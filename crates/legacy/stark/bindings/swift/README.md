# Stark

Stark is an independent fork of the Uzu inference engine with Rust, Metal, and CPU
implementation paths. This directory contains the inherited high-level API and
language bindings. The root README documents the scoring scope and current
validation evidence.

Stark binding packages have not been published. Build them from the repository
before using these examples. Upstream Mirai model identifiers and provider
endpoints remain in the inherited registry integration for compatibility.

## Quick Start



From the Stark repository root, build the local Swift artifact:

```sh
cargo tools install swift
cargo tools build swift --targets macos
```

Add the checkout as a local Swift package:

```swift
dependencies: [
    .package(path: "/path/to/stark")
]
```

The package requires `crates/legacy/stark/bindings/swift/stark.xcframework`.
Published Stark Swift binaries are not configured.

Run the code below:

```swift
import Foundation
import Stark

public func runQuickStart() async throws {
    let engineConfig = EngineConfig.create()
    let engine = try await Engine.create(config: engineConfig)
    
    guard let model = try await engine.model(identifier: "alibaba:qwen3.5:0.8b:mirai:mirai-m:4") else {
        return
    }
    
    for try await update in try await engine.download(model: model).iterator() {
        print(String(format: "\r\u{001B}[2KDownload progress: %.2f%%", update.progress() * 100), terminator: "")
        fflush(stdout)
    }
    print()
    
    let session = try await engine.chat(model: model, config: .create())
    
    let messages = [
        ChatMessage.system().withText(text: "You are a helpful assistant"),
        ChatMessage.user().withText(text: "Tell me a short, funny story about a robot")
    ]
    
    let reply = try await session.reply(input: messages, config: .create())
    guard let message = reply.last?.message else {
        return
    }
    
    print("Reasoning: \(message.reasoning() ?? "empty")")
    print("Text: \(message.text() ?? "empty")")
}
```


<br>

These examples use the inherited high-level registry and inference API. See the [repository README](https://github.com/jshahbazi/stark) for the supported scoring scope and validation evidence.

## Examples

You can run any example via `cargo tools example` \<**swift**\> \<**chat** | **chat-cloud** | **chat-shared-instance** | **chat-structured-output** | **quick-start** | **tool-calls**\>:

### Chat

In this example, we will download a model and get a reply to a specific list of messages:

```swift
import Foundation
import Stark

public func runChat() async throws {
    let engineConfig = EngineConfig.create()
    let engine = try await Engine.create(config: engineConfig)
    
    guard let model = try await engine.model(identifier: "alibaba:qwen3.5:0.8b:mirai:mirai-m:4") else {
        return
    }
    for try await update in try await engine.download(model: model).iterator() {
        print(String(format: "\r\u{001B}[2KDownload progress: %.2f%%", update.progress() * 100), terminator: "")
        fflush(stdout)
    }
    print()
    
    let messages = [
        ChatMessage.system().withText(text: "You are a helpful assistant"),
        ChatMessage.user().withText(text: "Tell me a short, funny story about a robot")
    ]
    let session = try await engine.chat(model: model, config: .create())
    let stream = await session.replyWithStream(input: messages, config: .create())
    var message: ChatMessage? = nil
    for try await update in stream.iterator() {
        switch update {
        case .replies(let replies):
            let reply = replies.last
            message = reply?.message
            print("Generated tokens: \(reply?.stats.tokensCountOutput ?? 0)")
        case .error(let error):
            print("Error: \(error)")
        }
    }
    print("Reasoning: \(message?.reasoning() ?? "empty")")
    print("Text: \(message?.text() ?? "empty")")
}
```

<br>Once loaded, the same `ChatSession` can be reused for multiple requests until you drop it. Each model may consume a significant amount of RAM, so it's important to keep only one session loaded at a time. For iOS apps, we recommend adding the [Increased Memory Capability](https://developer.apple.com/documentation/bundleresources/entitlements/com.apple.developer.kernel.increased-memory-limit) entitlement to ensure your app can allocate the required memory.

### Chat with the cloud model

In this example, we will get a reply to a specific list of messages from a cloud model:

```swift
import Stark

public func runChatCloud() async throws {
    let engineConfig = EngineConfig.create().withOpenaiApiKey(openaiApiKey: "OPENAI_API_KEY")
    let engine = try await Engine.create(config: engineConfig)
    
    guard let model = try await engine.model(identifier: "gpt-5") else {
        return
    }
    
    let messages = [
        ChatMessage.system().withReasoningEffort(reasoningEffort: .low),
        ChatMessage.user().withText(text: "How LLMs work")
    ]
    
    let session = try await engine.chat(model: model, config: .create())
    let reply = try await session.reply(input: messages, config: .create())
    guard let message = reply.last?.message else {
        return
    }
    
    print("Reasoning: \(message.reasoning() ?? "empty")")
    print("Text: \(message.text() ?? "empty")")
}
```

### Chat with shared instance

This example shows how to reuse chat instance without reloading model into memory:

```swift
import Foundation
import Stark

public func runChatSharedInstance() async throws {
    let engineConfig = EngineConfig.create()
    let engine = try await Engine.create(config: engineConfig)

    guard let model = try await engine.model(identifier: "alibaba:qwen3.5:0.8b:mirai:mirai-m:4") else {
        return
    }
    for try await update in try await engine.download(model: model).iterator() {
        print(String(format: "\r\u{001B}[2KDownload progress: %.2f%%", update.progress() * 100), terminator: "")
        fflush(stdout)
    }
    print()

    // The chatInstance owns the loaded model and can be shared between sessions.
    let chatInstance = try await engine.chatInstance(model: model, config: .create())

    let firstSession = try await engine.chatWithInstance(instance: chatInstance)
    let replies = try await firstSession.reply(
        input: [ChatMessage.user().withText(text: "Tell me a short, funny story about a robot")],
        config: .create()
    )
    if let message = replies.last?.message {
        print("First session reasoning: \(message.reasoning() ?? "")")
        print("First session text: \(message.text() ?? "")")
    }

    // The second session reuses the already-loaded weights instead of loading the model again.
    let secondSession = try await engine.chatWithInstance(instance: chatInstance)
    let secondReplies = try await secondSession.reply(
        input: [ChatMessage.user().withText(text: "What is the capital of France?")],
        config: .create()
    )
    if let message = secondReplies.last?.message {
        print("\nSecond session reasoning: \(message.reasoning() ?? "")")
        print("Second session text: \(message.text() ?? "")")
    }
}
```

### Chat with structured output

Sometimes you want the generated output to be valid JSON with predefined fields. You can use `Grammar` to manually specify a JSON schema for the response you want to receive:

```swift
import Foundation
import FoundationModels
import Stark

@Generable()
struct Country: Codable {
    let name: String
    let capital: String
}

public func runChatStructuredOutput() async throws {
    let engineConfig = EngineConfig.create()
    let engine = try await Engine.create(config: engineConfig)
    
    guard let model = try await engine.model(identifier: "alibaba:qwen3.5:0.8b:mirai:mirai-m:4") else {
        return
    }
    for try await update in try await engine.download(model: model).iterator() {
        print(String(format: "\r\u{001B}[2KDownload progress: %.2f%%", update.progress() * 100), terminator: "")
        fflush(stdout)
    }
    print()
    
    let messages = [
        ChatMessage.system().withReasoningEffort(reasoningEffort: .disabled),
        ChatMessage.user().withText(text: "Give me a JSON object containing a list of 3 countries, where each country has name and capital fields")
    ]
    
    let session = try await engine.chat(model: model, config: .create())
    let reply = try await session.reply(input: messages, config: .create().withGrammar(grammar: .fromType([Country].self)))
    guard let message = reply.last?.message else {
        return
    }
    guard let countries: [Country] = message.textDecoded() else {
        return
    }
    print(countries)
}
```

### Tool calls

This example shows how to use external tools:

```swift
import Foundation
import FoundationModels
import Stark

@Generable
private struct Coordinate: Codable, Sendable {
    @Guide(description: "Latitude in decimal degrees.")
    let latitude: Double

    @Guide(description: "Longitude in decimal degrees.")
    let longitude: Double
}

private struct GetCurrentLocation: Tool {
    let description = "Returns current location in coordinates"

    @Generable
    struct Arguments {
    }

    func call(arguments: Arguments) async throws -> Coordinate {
        Coordinate(latitude: 51.5074, longitude: -0.1278)
    }
}

private struct GetCurrentTemperature: Tool {
    let description = "Returns temperature in provided location"

    func call(arguments: Coordinate) async throws -> Double {
        _ = arguments
        return 25.0
    }
}

public func runToolCalls() async throws {
    let engine = try await Engine.create(config: .create())
    guard let model = try await engine.model(identifier: "alibaba:qwen3.5:0.8b:mirai:mirai-m:4") else {
        throw ToolCallsExampleError.modelNotFound
    }
    for try await update in try await engine.download(model: model).iterator() {
        print(String(format: "\r\u{001B}[2KDownload progress: %.2f%%", update.progress() * 100), terminator: "")
        fflush(stdout)
    }
    print()

    let session = try await engine.chat(model: model, config: .create())
    try await session.addTool(GetCurrentLocation())
    try await session.addTool(GetCurrentTemperature())

    let messages = [
        ChatMessage.system().withText(text: "You are a helpful assistant"),
        ChatMessage.user().withText(text: "What temperature is it now at my location?"),
    ]
    let reply_config = ChatReplyConfig.create().withSamplingMethod(samplingMethod: .greedy)
    let replies = try await session.reply(input: messages, config: reply_config)
    guard let message = replies.last?.message else {
        return
    }

    print("Reasoning: \(message.reasoning() ?? "")")
    print("Text: \(message.text() ?? "")")
}

private enum ToolCallsExampleError: Swift.Error {
    case modelNotFound
}
```



## Troubleshooting

Report Stark problems in the [issue tracker](https://github.com/jshahbazi/stark/issues).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
