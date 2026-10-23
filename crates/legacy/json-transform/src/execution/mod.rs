mod access;
mod collection;
mod condition;
mod construction;
mod control_flow;
mod operation;
mod string;

pub use condition::Condition;
pub use operation::{CallTarget, GetTarget, Operation, PathSegment, Pipeline, SwitchCase};
use serde_json::Value;

use crate::{TransformError, TransformSchema};

pub(crate) const MAX_DEPTH: usize = 32;
const MAX_WORK: usize = 65_536;

pub(crate) struct ExecutionContext {
    depth: usize,
    remaining: usize,
}

impl Default for ExecutionContext {
    fn default() -> Self {
        Self {
            depth: 0,
            remaining: MAX_WORK,
        }
    }
}

impl ExecutionContext {
    fn spend(&mut self) -> Result<(), TransformError> {
        self.remaining = self.remaining.checked_sub(1).ok_or(TransformError::ExecutionLimit)?;
        Ok(())
    }
}

pub(crate) fn execute_pipeline(
    pipeline: &Pipeline,
    input: Value,
    schema: &TransformSchema,
    context: &mut ExecutionContext,
) -> Result<Value, TransformError> {
    context.spend()?;
    if context.depth >= MAX_DEPTH {
        return Err(TransformError::ExecutionLimit);
    }
    context.depth += 1;
    let result = (|| {
        let mut output = input;
        for operation in pipeline {
            output = operation.execute_with_context(output, schema, context)?;
        }
        Ok(output)
    })();
    context.depth -= 1;
    result
}
