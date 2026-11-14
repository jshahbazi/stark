use num_traits::Float;

const GELU_K0: f32 = 0.044715;
const GELU_K1: f32 = 0.7978846; // sqrt(2/pi)

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum ActivationType {
    SILU,
    GELUApprox,
    GELUExact,
    IDENTITY,
    SOFTPLUS,
    SILUMaterializedSigmoid,
    GELUMaterializedPolynomial,
}

impl ActivationType {
    pub fn activate<T: Float>(
        &self,
        x: T,
    ) -> T {
        match self {
            ActivationType::SILU => silu(x),
            ActivationType::SILUMaterializedSigmoid => silu_materialized_sigmoid(x),
            ActivationType::GELUApprox => gelu_approx(x),
            ActivationType::GELUMaterializedPolynomial => gelu_materialized_polynomial(x),
            ActivationType::GELUExact => gelu_exact(x),
            ActivationType::IDENTITY => x,
            ActivationType::SOFTPLUS => softplus(x),
        }
    }
}

pub fn activation_silu_alpha<T: Float>(
    x: T,
    alpha: f32,
) -> T {
    let x_float = x.to_f32().unwrap();
    let y_float = x_float / (1.0f32 + (-alpha * x_float).exp());
    T::from(y_float).unwrap()
}

fn silu<T: Float>(x: T) -> T {
    activation_silu_alpha(x, 1.0f32)
}

fn silu_materialized_sigmoid<T: Float>(x: T) -> T {
    // MLX v0.32.3 unary_ops.h Sigmoid and bf16_math.h materialize the
    // exponential and every subsequent BF16 scalar operation.
    let one = T::one();
    let exponential = T::from(x.to_f32().unwrap().abs().exp()).unwrap();
    let lower = one / (one + exponential);
    let sigmoid = if x < T::zero() {
        lower
    } else {
        one - lower
    };
    x * sigmoid
}

fn gelu_approx<T: Float>(x: T) -> T {
    let x_float = x.to_f32().unwrap();
    let tan_arg = GELU_K1 * (x_float + GELU_K0 * x_float * x_float * x_float);
    let y_float = 0.5f32 * x_float * (1.0f32 + tan_arg.tanh());
    T::from(y_float).unwrap()
}

fn gelu_materialized_polynomial<T: Float>(x: T) -> T {
    let cubic = T::from(x.to_f32().unwrap().powi(3)).unwrap();
    let term = T::from(GELU_K0).unwrap() * cubic;
    let inner = x + term;
    let argument = T::from(GELU_K1).unwrap() * inner;
    let activated = T::from(argument.to_f32().unwrap().tanh()).unwrap();
    (T::from(0.5).unwrap() * x) * (T::one() + activated)
}

fn gelu_exact<T: Float>(x: T) -> T {
    let x_float = x.to_f32().unwrap();
    let y_float = 0.5f32 * x_float * (1.0f32 + libm::erff(x_float * std::f32::consts::FRAC_1_SQRT_2));
    T::from(y_float).unwrap()
}

fn softplus<T: Float>(x: T) -> T {
    let x_float = x.to_f32().unwrap();
    if x_float > 20f32 {
        return x;
    }

    let result = (1f32 + x_float.exp()).ln();
    T::from(result).unwrap()
}
