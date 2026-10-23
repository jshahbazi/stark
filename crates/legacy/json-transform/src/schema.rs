use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{TransformError, execution::Pipeline};

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(transparent)]
pub struct TransformSchema {
    pub pipelines: IndexMap<String, Pipeline>,
}

impl TransformSchema {
    pub fn execute(
        &self,
        name: &str,
        input: Value,
    ) -> Result<Value, TransformError> {
        self.validate()?;
        let pipeline = self.pipelines.get(name).ok_or_else(|| TransformError::UndefinedPipeline {
            name: name.to_string(),
        })?;
        crate::execution::execute_pipeline(pipeline, input, self, &mut crate::execution::ExecutionContext::default())
    }

    /// Reject static cycles and oversized syntax before any recursive execution.
    pub fn validate(&self) -> Result<(), TransformError> {
        use std::collections::{HashMap, HashSet};

        use crate::execution::{CallTarget, MAX_DEPTH, Operation};

        let mut graph: HashMap<&str, Vec<&str>> = HashMap::new();
        let mut work = 0usize;
        for (name, pipeline) in &self.pipelines {
            let mut calls = Vec::new();
            let mut pending: Vec<_> = pipeline.iter().map(|op| (op, 1usize)).collect();
            while let Some((operation, depth)) = pending.pop() {
                work += 1;
                if work > 65_536 || depth > MAX_DEPTH {
                    return Err(TransformError::ExecutionLimit);
                }
                let mut children: Vec<&Pipeline> = Vec::new();
                match operation {
                    Operation::Call {
                        target: CallTarget::Static {
                            name,
                        },
                        ..
                    } => calls.push(name.as_str()),
                    Operation::Object {
                        fields,
                        ..
                    } => children.extend(fields.values()),
                    Operation::Each {
                        apply,
                    }
                    | Operation::FlatMap {
                        apply,
                    } => children.push(apply),
                    Operation::Reduce {
                        key,
                        then,
                        ..
                    } => children.extend([key, then]),
                    Operation::Switch {
                        key,
                        cases,
                        default,
                    } => {
                        children.push(key);
                        children.extend(cases.iter().map(|case| &case.then));
                        children.extend(default.iter());
                    },
                    Operation::On {
                        r#do,
                        ..
                    } => children.push(r#do),
                    _ => {},
                }
                for child in children {
                    pending.extend(child.iter().map(|op| (op, depth + 1)));
                }
            }
            graph.insert(name, calls);
        }
        let mut done = HashSet::new();
        let mut active = HashSet::new();
        for root in graph.keys().copied() {
            let mut pending = vec![(root, false)];
            while let Some((name, exiting)) = pending.pop() {
                if exiting {
                    active.remove(name);
                    done.insert(name);
                } else if !done.contains(name) {
                    if !active.insert(name) {
                        return Err(TransformError::CyclicPipeline);
                    }
                    pending.push((name, true));
                    if let Some(calls) = graph.get(name) {
                        pending.extend(calls.iter().map(|target| (*target, false)));
                    }
                }
            }
        }
        Ok(())
    }
}
