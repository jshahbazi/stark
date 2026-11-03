use std::fmt::Write as _;

use minijinja::{
    AutoEscape, Environment, Error, ErrorKind, Value, filters, functions,
    machinery::{Token, tokenize},
    value::{Kwargs, Rest, ValueKind, from_args},
};
use serde::Deserialize;
use serde_json::Value as Json;

pub const MAX_REQUEST_BYTES: usize = 2 * 1024 * 1024;
const MAX_SOURCE_BYTES: usize = 256 * 1024;
const MAX_OUTPUT_BYTES: usize = 1024 * 1024;
const MAX_ITEMS: usize = 16384;
const MAX_NODES: usize = 65536;
const MAX_DEPTH: usize = 64;
const MAX_TOKENS: usize = 16384;
const FUEL: u64 = 1_000_000;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Failure {
    Runtime,
    Sandbox,
    Size,
}

impl Failure {
    pub fn status(self) -> u8 {
        match self {
            Self::Runtime => b'E',
            Self::Sandbox => b'S',
            Self::Size => b'L',
        }
    }
}

fn unsupported() -> Error {
    Error::from(ErrorKind::InvalidOperation)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    source: String,
    arguments: Option<Json>,
}

fn check_json(
    value: &Json,
    depth: usize,
    nodes: &mut usize,
    size: &mut usize,
) -> Result<(), Failure> {
    *nodes += 1;
    if *nodes > MAX_NODES || depth > MAX_DEPTH {
        return Err(Failure::Runtime);
    }
    match value {
        Json::String(value) => *size += value.len(),
        Json::Array(items) => {
            if items.len() > MAX_ITEMS {
                return Err(Failure::Runtime);
            }
            *size += items.len() * 4;
            for item in items {
                check_json(item, depth + 1, nodes, size)?;
            }
        },
        Json::Object(items) => {
            if items.len() > MAX_ITEMS {
                return Err(Failure::Runtime);
            }
            *size += items.len() * 4;
            for (key, item) in items {
                *nodes += 1;
                *size += key.len();
                check_json(item, depth + 1, nodes, size)?;
            }
        },
        _ => *size += 32,
    }
    if *size > MAX_OUTPUT_BYTES {
        return Err(Failure::Size);
    }
    Ok(())
}

fn membership_tuple_end(
    tokens: &[(Token<'_>, minijinja::machinery::Span)],
    start: usize,
) -> Option<usize> {
    if start == 0 || !matches!(tokens[start - 1].0, Token::Ident("in")) {
        return None;
    }
    let mut expecting_string = true;
    let mut seen_string = false;
    let mut seen_comma = false;
    for (index, (token, _)) in tokens.iter().enumerate().skip(start + 1) {
        match token {
            Token::Str(_) | Token::String(_) if expecting_string => {
                expecting_string = false;
                seen_string = true;
            },
            Token::Comma if !expecting_string => {
                expecting_string = true;
                seen_comma = true;
            },
            Token::ParenClose if seen_string && seen_comma => {
                // Only the boolean result may escape. In particular, chained
                // comparisons could reuse the container operand itself.
                return match tokens.get(index + 1).map(|(token, _)| token) {
                    Some(Token::BlockEnd | Token::VariableEnd | Token::ParenClose | Token::Ident("and" | "or")) => {
                        Some(index)
                    },
                    _ => None,
                };
            },
            _ => return None,
        }
    }
    None
}

fn check_source(source: &str) -> Result<(), Failure> {
    if source.is_empty() || source.len() > MAX_SOURCE_BYTES {
        return Err(Failure::Runtime);
    }
    // MiniJinja represents tuple literals as lists. Reject that unsupported
    // construct before rendering instead of silently changing Python repr/type
    // behavior. Constant-string tuples used solely as a membership RHS have
    // identical primitive membership semantics and are admitted narrowly.
    // Parenthesized calls and groups remain supported; comma-separated
    // loop/assignment targets need no parentheses.
    let tokens = tokenize(source, false, Default::default(), Default::default())
        .take(MAX_TOKENS + 1)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| Failure::Runtime)?;
    if tokens.len() > MAX_TOKENS {
        return Err(Failure::Runtime);
    }
    let mut nesting = Vec::new();
    let mut previous_callable = false;
    let mut statement_start = false;
    let mut set_statement = false;
    let mut set_rhs = false;
    let mut condition_statement = false;
    let mut value_expression = false;
    for (index, (token, _)) in tokens.iter().enumerate() {
        if statement_start {
            set_statement = matches!(token, Token::Ident("set"));
            condition_statement = matches!(token, Token::Ident("if" | "elif"));
            statement_start = false;
        }
        match token {
            Token::BlockStart => {
                statement_start = true;
                set_statement = false;
                set_rhs = false;
                condition_statement = false;
            },
            Token::BlockEnd => {
                set_statement = false;
                set_rhs = false;
                condition_statement = false;
            },
            Token::VariableStart => value_expression = true,
            Token::VariableEnd => value_expression = false,
            Token::Assign if set_statement && nesting.is_empty() => set_rhs = true,
            // A set RHS also accepts bare tuples, including a single value
            // followed by a comma. Target-unpacking commas precede Assign.
            Token::Comma if set_rhs && nesting.is_empty() => return Err(Failure::Runtime),
            Token::Ident(name) if name.starts_with('_') => return Err(Failure::Sandbox),
            Token::Ident("autoescape" | "include" | "import" | "from" | "extends") => return Err(Failure::Sandbox),
            Token::ParenOpen | Token::BracketOpen | Token::BraceOpen => {
                let membership_only = matches!(token, Token::ParenOpen)
                    && (condition_statement || set_rhs || value_expression)
                    && membership_tuple_end(&tokens, index).is_some();
                nesting.push((matches!(token, Token::ParenOpen), previous_callable, index, membership_only));
                if nesting.len() > 128 {
                    return Err(Failure::Runtime);
                }
            },
            Token::Comma
                if nesting
                    .last()
                    .is_some_and(|&(parenthesis, call, _, membership)| parenthesis && !call && !membership) =>
            {
                return Err(Failure::Runtime);
            },
            Token::BracketClose | Token::ParenClose | Token::BraceClose => {
                let (parenthesis, call, opened_at, _) = nesting.pop().ok_or(Failure::Runtime)?;
                if parenthesis && !call && index == opened_at + 1 {
                    return Err(Failure::Runtime);
                }
            },
            _ => {},
        }
        previous_callable = match token {
            Token::Ident(name) => !matches!(
                *name,
                "if" | "elif" | "else" | "for" | "in" | "is" | "not" | "and" | "or" | "set" | "do" | "with"
            ),
            Token::ParenClose | Token::BracketClose | Token::BraceClose => true,
            _ => false,
        };
    }
    Ok(())
}

/// serde_json otherwise converts integers outside u64/i64 into approximate
/// floats. Reject unsupported integer inputs before that lossy conversion.
fn check_integer_tokens(data: &[u8]) -> Result<(), Failure> {
    let mut quoted = false;
    let mut escaped = false;
    let mut index = 0;
    while index < data.len() {
        let byte = data[index];
        if quoted {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                quoted = false;
            }
            index += 1;
        } else if byte == b'"' {
            quoted = true;
            index += 1;
        } else if byte == b'-' || byte.is_ascii_digit() {
            let start = index;
            while index < data.len() && matches!(data[index], b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E') {
                index += 1;
            }
            let token = &data[start..index];
            if !token.iter().any(|byte| matches!(byte, b'.' | b'e' | b'E')) {
                let text = std::str::from_utf8(token).map_err(|_| Failure::Runtime)?;
                if text.parse::<i64>().is_err() && text.parse::<u64>().is_err() {
                    return Err(Failure::Runtime);
                }
            }
        } else {
            index += 1;
        }
    }
    Ok(())
}

fn bounded_range(arguments: Rest<i64>) -> Result<Value, Error> {
    let (start, stop, step) = match arguments.as_slice() {
        [stop] => (0i128, *stop as i128, 1i128),
        [start, stop] => (*start as i128, *stop as i128, 1),
        [start, stop, step] if *step != 0 => (*start as i128, *stop as i128, *step as i128),
        _ => return Err(unsupported()),
    };
    let length = if (step > 0 && stop <= start) || (step < 0 && stop >= start) {
        0
    } else {
        usize::try_from(((stop - start).abs() + step.abs() - 1) / step.abs()).map_err(|_| unsupported())?
    };
    if length > MAX_ITEMS {
        return Err(unsupported());
    }
    Ok((0..length).map(|index| Value::from((start + index as i128 * step) as i64)).collect())
}

fn python_float(value: f64) -> Result<String, Error> {
    if !value.is_finite() {
        return Err(unsupported());
    }
    let text = format!("{value:?}");
    if let Some((mantissa, exponent)) = text.split_once('e') {
        let exponent: i32 = exponent.parse().map_err(|_| unsupported())?;
        Ok(format!("{mantissa}e{exponent:+03}"))
    } else {
        Ok(text)
    }
}

fn check_serializable(
    value: &Value,
    depth: usize,
    nodes: &mut usize,
) -> Result<(), Error> {
    *nodes += 1;
    if depth > MAX_DEPTH || *nodes > MAX_NODES {
        return Err(unsupported());
    }
    match value.kind() {
        ValueKind::Number => {
            if !f64::try_from(value.clone())?.is_finite() {
                return Err(unsupported());
            }
        },
        ValueKind::String | ValueKind::Bool | ValueKind::None => {},
        ValueKind::Seq | ValueKind::Map => {
            if value.len().is_some_and(|length| length > MAX_ITEMS) {
                return Err(unsupported());
            }
            for (index, child) in value.try_iter()?.enumerate() {
                if index >= MAX_ITEMS {
                    return Err(unsupported());
                }
                check_serializable(&child, depth + 1, nodes)?;
                if value.kind() == ValueKind::Map {
                    check_serializable(&value.get_item(&child)?, depth + 1, nodes)?;
                }
            }
        },
        _ => return Err(unsupported()),
    }
    Ok(())
}

fn quoted(
    value: &str,
    json: bool,
) -> String {
    let quote = if json || (value.contains('\'') && !value.contains('"')) {
        '"'
    } else {
        '\''
    };
    let mut result = String::new();
    result.push(quote);
    for character in value.chars() {
        match character {
            '\\' => result.push_str("\\\\"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            '\u{8}' if json => result.push_str("\\b"),
            '\u{c}' if json => result.push_str("\\f"),
            character if character == quote => {
                result.push('\\');
                result.push(character);
            },
            '<' | '>' | '&' | '\'' if json => {
                let _ = write!(result, "\\u{:04x}", character as u32);
            },
            character if json && (!character.is_ascii() || character.is_control()) => {
                for unit in character.encode_utf16(&mut [0u16; 2]) {
                    let _ = write!(result, "\\u{unit:04x}");
                }
            },
            character if character.is_control() || character.escape_debug().count() > 1 => {
                if character as u32 <= 255 {
                    let _ = write!(result, "\\x{:02x}", character as u32);
                } else if character as u32 <= 0xffff {
                    let _ = write!(result, "\\u{:04x}", character as u32);
                } else {
                    let _ = write!(result, "\\U{:08x}", character as u32);
                }
            },
            character => result.push(character),
        }
    }
    result.push(quote);
    result
}

fn python_repr(
    value: &Value,
    nested: bool,
    depth: usize,
    nodes: &mut usize,
) -> Result<String, Error> {
    *nodes += 1;
    if depth > MAX_DEPTH || *nodes > MAX_NODES {
        return Err(unsupported());
    }
    let result = match value.kind() {
        ValueKind::Undefined => {
            if nested {
                "Undefined".to_owned()
            } else {
                String::new()
            }
        },
        ValueKind::None => "None".to_owned(),
        ValueKind::Bool => {
            if value.is_true() {
                "True".to_owned()
            } else {
                "False".to_owned()
            }
        },
        ValueKind::String => {
            let text = value.as_str().ok_or_else(unsupported)?;
            if nested {
                quoted(text, false)
            } else {
                text.to_owned()
            }
        },
        ValueKind::Number => {
            let json = serde_json::to_value(value).map_err(|_| unsupported())?;
            match json {
                Json::Number(number) if number.is_f64() => python_float(number.as_f64().ok_or_else(unsupported)?)?,
                Json::Number(number) => number.to_string(),
                _ => return Err(unsupported()),
            }
        },
        ValueKind::Seq | ValueKind::Iterable | ValueKind::Map => {
            if value.len().is_some_and(|length| length > MAX_ITEMS) {
                return Err(unsupported());
            }
            let mapping = value.kind() == ValueKind::Map;
            let mut result = String::from(if mapping {
                "{"
            } else {
                "["
            });
            for (index, item) in value.try_iter()?.enumerate() {
                if index >= MAX_ITEMS {
                    return Err(unsupported());
                }
                if index > 0 {
                    result.push_str(", ");
                }
                result.push_str(&python_repr(&item, true, depth + 1, nodes)?);
                if mapping {
                    result.push_str(": ");
                    result.push_str(&python_repr(&value.get_item(&item)?, true, depth + 1, nodes)?);
                }
                if result.len() > MAX_OUTPUT_BYTES {
                    return Err(unsupported());
                }
            }
            result.push(if mapping {
                '}'
            } else {
                ']'
            });
            result
        },
        _ => return Err(unsupported()),
    };
    if result.len() > MAX_OUTPUT_BYTES {
        return Err(unsupported());
    }
    Ok(result)
}

fn write_json(
    value: &Json,
    output: &mut String,
    indent: Option<usize>,
    depth: usize,
) -> Result<(), Error> {
    if depth > MAX_DEPTH || output.len() > MAX_OUTPUT_BYTES {
        return Err(unsupported());
    }
    let newline = |output: &mut String, depth: usize| {
        if let Some(indent) = indent {
            output.push('\n');
            for _ in 0..depth * indent {
                output.push(' ');
            }
        }
    };
    match value {
        Json::String(value) => output.push_str(&quoted(value, true)),
        Json::Number(value) if value.is_f64() => {
            output.push_str(&python_float(value.as_f64().ok_or_else(unsupported)?)?)
        },
        Json::Number(value) => {
            let _ = write!(output, "{value}");
        },
        Json::Bool(value) => output.push_str(if *value {
            "true"
        } else {
            "false"
        }),
        Json::Null => output.push_str("null"),
        Json::Array(items) => {
            if items.len() > MAX_ITEMS {
                return Err(unsupported());
            }
            output.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    output.push(',');
                    if indent.is_none() {
                        output.push(' ');
                    }
                }
                newline(output, depth + 1);
                write_json(item, output, indent, depth + 1)?;
            }
            if !items.is_empty() {
                newline(output, depth);
            }
            output.push(']');
        },
        Json::Object(items) => {
            if items.len() > MAX_ITEMS {
                return Err(unsupported());
            }
            let mut items: Vec<_> = items.iter().collect();
            items.sort_unstable_by(|left, right| left.0.cmp(right.0));
            output.push('{');
            for (index, (key, item)) in items.iter().enumerate() {
                if index > 0 {
                    output.push(',');
                    if indent.is_none() {
                        output.push(' ');
                    }
                }
                newline(output, depth + 1);
                output.push_str(&quoted(key, true));
                output.push_str(": ");
                write_json(item, output, indent, depth + 1)?;
            }
            if !items.is_empty() {
                newline(output, depth);
            }
            output.push('}');
        },
    }
    if output.len() > MAX_OUTPUT_BYTES {
        return Err(unsupported());
    }
    Ok(())
}

fn tojson(
    value: &Value,
    positional: Option<Value>,
    kwargs: Kwargs,
) -> Result<Value, Error> {
    let named: Option<Value> = kwargs.get("indent")?;
    kwargs.assert_all_used()?;
    let indent = positional.or(named).filter(|value| !value.is_none());
    let indent = match indent {
        None => None,
        Some(value) => {
            let indent = value.as_usize().filter(|indent| *indent <= 16).ok_or_else(unsupported)?;
            Some(indent)
        },
    };
    check_serializable(value, 0, &mut 0)?;
    let value = serde_json::to_value(value).map_err(|_| unsupported())?;
    check_json(&value, 0, &mut 0, &mut 0).map_err(|_| unsupported())?;
    let mut result = String::new();
    write_json(&value, &mut result, indent, 0)?;
    Ok(Value::from(result))
}

fn methods(
    state: &minijinja::State,
    value: &Value,
    method: &str,
    args: &[Value],
) -> Result<Value, Error> {
    let allowed = match value.kind() {
        ValueKind::String => matches!(
            method,
            "startswith"
                | "endswith"
                | "strip"
                | "lstrip"
                | "rstrip"
                | "split"
                | "lower"
                | "upper"
                | "capitalize"
                | "title"
                | "count"
                | "find"
                | "rfind"
                | "isdigit"
                | "isalpha"
                | "isalnum"
                | "isspace"
                | "islower"
                | "isupper"
                | "isnumeric"
        ),
        ValueKind::Map => matches!(method, "get" | "items" | "keys" | "values"),
        ValueKind::Seq => method == "count",
        _ => false,
    };
    if !allowed {
        return Err(unsupported());
    }
    if let Some(text) = value.as_str() {
        if matches!(method, "strip" | "lstrip" | "rstrip") {
            let (characters,): (Option<&str>,) = from_args(args)?;
            let predicate = |character: char| match characters {
                Some(characters) => characters.contains(character),
                None => character.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&character),
            };
            let result = match method {
                "lstrip" => text.trim_start_matches(predicate),
                "rstrip" => text.trim_end_matches(predicate),
                _ => text.trim_matches(predicate),
            };
            return Ok(Value::from(result));
        }
        if matches!(method, "count" | "find" | "rfind") {
            let (needle,): (&str,) = from_args(args)?;
            let result = match method {
                "count" => {
                    (if needle.is_empty() {
                        text.chars().count() + 1
                    } else {
                        text.matches(needle).count()
                    }) as i64
                },
                "find" => text.find(needle).map(|index| text[..index].chars().count() as i64).unwrap_or(-1),
                _ => text.rfind(needle).map(|index| text[..index].chars().count() as i64).unwrap_or(-1),
            };
            return Ok(Value::from(result));
        }
        // These classifications differ across Python/Rust Unicode tables. Fail
        // closed outside ASCII rather than silently changing a model prompt.
        if matches!(method, "isdigit" | "isalpha" | "isalnum" | "isspace" | "islower" | "isupper" | "isnumeric") {
            let (): () = from_args(args)?;
            if !text.is_ascii() {
                return Err(unsupported());
            }
            let result = match method {
                "isdigit" | "isnumeric" => !text.is_empty() && text.bytes().all(|byte| byte.is_ascii_digit()),
                "isalpha" => !text.is_empty() && text.bytes().all(|byte| byte.is_ascii_alphabetic()),
                "isalnum" => !text.is_empty() && text.bytes().all(|byte| byte.is_ascii_alphanumeric()),
                "isspace" => {
                    !text.is_empty()
                        && text.bytes().all(|byte| byte.is_ascii_whitespace() || (0x1c..=0x1f).contains(&byte))
                },
                "islower" => {
                    text.bytes().any(|byte| byte.is_ascii_lowercase())
                        && !text.bytes().any(|byte| byte.is_ascii_uppercase())
                },
                _ => {
                    text.bytes().any(|byte| byte.is_ascii_uppercase())
                        && !text.bytes().any(|byte| byte.is_ascii_lowercase())
                },
            };
            return Ok(Value::from(result));
        }
    }
    minijinja_contrib::pycompat::unknown_method_callback(state, value, method, args)
}

fn environment<'source>() -> Environment<'source> {
    let mut environment = Environment::new();
    let globals: Vec<_> = environment.globals().map(|(name, _)| name.to_owned()).collect();
    for name in globals {
        environment.remove_global(&name);
    }
    environment.add_function("range", bounded_range);
    environment.add_function("dict", functions::dict);
    environment.add_function("namespace", functions::namespace);
    environment.add_function("raise_exception", |_value: Value| -> Result<Value, Error> { Err(unsupported()) });
    for filter in [
        "attr",
        "batch",
        "bool",
        "chain",
        "format",
        "groupby",
        "indent",
        "lines",
        "pprint",
        "slice",
        "split",
        "sum",
        "urlencode",
        "wordwrap",
        "zip",
    ] {
        environment.remove_filter(filter);
    }
    environment.add_filter("tojson", tojson);
    environment.add_filter("string", |value: &Value| python_repr(value, false, 0, &mut 0));
    environment.add_filter("forceescape", filters::escape);
    environment.set_unknown_method_callback(methods);
    environment.set_formatter(|output, _state, value| {
        output.write_str(&python_repr(value, false, 0, &mut 0)?).map_err(|_| unsupported())
    });
    environment.set_auto_escape_callback(|_| AutoEscape::None);
    environment.set_trim_blocks(true);
    environment.set_lstrip_blocks(true);
    environment.set_recursion_limit(128);
    environment.set_fuel(Some(FUEL));
    environment
}

#[derive(Default)]
struct Output {
    bytes: Vec<u8>,
    too_large: bool,
}

impl std::io::Write for Output {
    fn write(
        &mut self,
        bytes: &[u8],
    ) -> std::io::Result<usize> {
        if bytes.len() > MAX_OUTPUT_BYTES.saturating_sub(self.bytes.len()) {
            self.too_large = true;
            return Err(std::io::ErrorKind::OutOfMemory.into());
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

pub fn evaluate(data: &[u8]) -> Result<Vec<u8>, Failure> {
    if data.len() > MAX_REQUEST_BYTES {
        return Err(Failure::Size);
    }
    check_integer_tokens(data)?;
    let request: Request = serde_json::from_slice(data).map_err(|_| Failure::Runtime)?;
    check_source(&request.source)?;
    if let Some(arguments) = &request.arguments {
        if !arguments.is_object() {
            return Err(Failure::Runtime);
        }
        check_json(arguments, 0, &mut 0, &mut 0)?;
    }
    let mut environment = environment();
    environment.add_template("model", &request.source).map_err(|_| Failure::Runtime)?;
    let template = environment.get_template("model").map_err(|_| Failure::Runtime)?;
    let mut output = Output::default();
    if let Some(arguments) = request.arguments
        && template.render_captured_to(arguments, &mut output).is_err()
    {
        return Err(if output.too_large {
            Failure::Size
        } else {
            Failure::Runtime
        });
    }
    Ok(output.bytes)
}
