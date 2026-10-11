use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{ItemFn, Meta, parse_macro_input};

pub fn stark_test(
    _args: TokenStream,
    input: TokenStream,
) -> TokenStream {
    let mut func = parse_macro_input!(input as ItemFn);

    let ignore_attr =
        func.attrs.iter().position(|attr| attr.path().is_ident("ignore")).map(|index| func.attrs.remove(index));
    let ignore = match ignore_attr.as_ref().map(|attr| &attr.meta) {
        None => quote! { false },
        Some(Meta::Path(_)) => quote! { true },
        Some(meta) => {
            return syn::Error::new_spanned(meta, "unsupported #[ignore] form").to_compile_error().into();
        },
    };

    let name = &func.sig.ident;
    let const_name = format_ident!("__STARK_TEST_CASE_{}", name);

    quote! {
        #func

        #[test_case]
        #[allow(non_upper_case_globals)]
        const #const_name: crate::tests::harness::StarkTest =
            crate::tests::harness::StarkTest::Test(&crate::tests::harness::StarkTestCase {
                name: concat!(
                    module_path!(),
                    "::",
                    stringify!(#name),
                ),
                ignore: #ignore,
                run: #name,
            });
    }
    .into()
}

pub fn stark_bench(
    _args: TokenStream,
    input: TokenStream,
) -> TokenStream {
    let func = parse_macro_input!(input as ItemFn);

    let name = &func.sig.ident;
    let const_name = format_ident!("__STARK_BENCH_CASE_{}", name);

    quote! {
        #func

        #[test_case]
        #[allow(non_upper_case_globals)]
        const #const_name: crate::tests::harness::StarkTest = crate::tests::harness::StarkTest::Bench(&|| {
            let mut criterion = ::criterion::Criterion::default().configure_from_args();
            #name(&mut criterion);
        });
    }
    .into()
}
