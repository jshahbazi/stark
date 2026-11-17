# Perplexity Decider reference provenance

Source: https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.3

Revision: `d06fb3f2d97e0a52325a0d463c450cf0338f44fe`.

Source `protocol.py` SHA-256:
`412292fba7f9e92f70f3071935f090e381910d1f6279380e728dc1c6b8380e6c`.

`decider_protocol.py` adapts `source/src/pplx_decider/protocol.py` from this
revision. The port uses Python lists instead of Torch tensors, delegates chat
template evaluation to Stark's native template helper, delegates image decoding
and token expansion to the native Decider worker, and supports only the
reference's default separate-question layout. Error messages omit request
content. Locate and isolated-question execution are explicitly unavailable.

`LICENSE.txt` and `NOTICE.txt` reproduce the checkpoint's original license and
notice without changes. The license retains its Alibaba Cloud copyright notice.
Prepared artifacts also retain the checkpoint's original LICENSE and NOTICE.

This provenance does not establish numerical or performance qualification.
