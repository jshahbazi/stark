import Darwin
import Foundation

/// Uses OpenWork's authenticated, loopback UI-control API. No inference or tools
/// run here. The task is new, so an existing conversation is never rebound.
@MainActor final class OpenWorkBeginnerHandoff {
    enum Failure: Error { case unavailable, invalidReply }
    typealias Request = (String, [String: Any]?) async throws -> [String: Any]
    private let request: Request
    private(set) var sessionID: String?
    private var workspaceID: String?

    init(request: @escaping Request) { self.request = request }

    convenience init(discoveryURL: URL) throws {
        // The official app writes this file in its own user-data directory.
        // Never follow a replacement symlink or accept a foreign-owned file.
        let descriptor = open(discoveryURL.path, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC)
        guard descriptor >= 0 else { throw Failure.unavailable }
        defer { close(descriptor) }
        var info = stat()
        guard fstat(descriptor, &info) == 0, info.st_uid == geteuid(),
              info.st_mode & S_IFMT == S_IFREG, info.st_mode & 0o022 == 0,
              info.st_nlink == 1, info.st_size > 0, info.st_size <= 16_384 else { throw Failure.unavailable }
        let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: false)
        let data = handle.readData(ofLength: 16_385)
        guard data.count <= 16_384,
              let value = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              value["version"] as? Int == 2,
              value["identifier"] as? String == "com.differentai.openwork",
              let base = value["baseUrl"] as? String, let parts = URLComponents(string: base),
              parts.scheme == "http", parts.host == "127.0.0.1", let port = parts.port,
              (1...65535).contains(port), base == "http://127.0.0.1:\(port)",
              let token = value["token"] as? String, token.utf8.count == 64,
              token.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }) else { throw Failure.unavailable }
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 15
        configuration.httpShouldSetCookies = false
        configuration.connectionProxyDictionary = [:]
        let session = URLSession(configuration: configuration, delegate: NoRedirects(), delegateQueue: nil)
        self.init { path, body in
            var request = URLRequest(url: URL(string: base + path)!)
            request.setValue("Bearer " + token, forHTTPHeaderField: "Authorization")
            if let body {
                request.httpMethod = "POST"
                request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                request.httpBody = try JSONSerialization.data(withJSONObject: body)
            }
            let (data, response) = try await session.data(for: request)
            guard (response as? HTTPURLResponse)?.statusCode == 200, data.count <= 1_048_576,
                  let result = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                  result["ok"] as? Bool == true else { throw Failure.invalidReply }
            return result
        }
    }

    func prepare(modelID: String, isCurrent: () -> Bool = { true }) async throws {
        guard isCurrent() else { throw Failure.unavailable }
        let context = try await request("/context", nil)
        guard let screen = (context["context"] as? [String: Any])?["screen"] as? [String: Any],
              let workspace = screen["workspaceId"] as? String, !workspace.isEmpty else { throw Failure.unavailable }
        let catalog = try await action("/query", "models.list", ["workspaceId": workspace])
        guard let models = catalog["models"] as? [[String: Any]], models.contains(where: {
            $0["providerId"] as? String == "stark" && $0["modelId"] as? String == modelID && $0["available"] as? Bool == true
        }) else { throw Failure.unavailable }
        guard isCurrent(), workspaceID == nil || workspaceID == workspace else { throw Failure.unavailable }
        workspaceID = workspace
        if sessionID == nil {
            let current = try await request("/context", nil)
            guard isCurrent(), ((current["context"] as? [String: Any])?["screen"] as? [String: Any])?["workspaceId"] as? String == workspace else { throw Failure.unavailable }
            let created = try await request("/command", ["id": "session.create_task", "args": [:]])
            guard created["ok"] as? Bool == true, let id = created["result"] as? String, !id.isEmpty else { throw Failure.invalidReply }
            sessionID = id
        }
        guard isCurrent(), let sessionID else { throw Failure.invalidReply }
        let bound = try await action("/command", "session.set_model", [
            "workspaceId": workspace, "sessionId": sessionID, "providerId": "stark", "modelId": modelID,
        ])
        guard bound["ok"] as? Bool == true, bound["sessionId"] as? String == sessionID,
              bound["workspaceId"] as? String == workspace,
              let model = bound["model"] as? [String: Any], model["providerId"] as? String == "stark",
              model["modelId"] as? String == modelID else { throw Failure.invalidReply }
        let final = try await request("/context", nil)
        guard let selected = (final["context"] as? [String: Any])?["screen"] as? [String: Any],
              selected["workspaceId"] as? String == workspace,
              selected["sessionId"] as? String == sessionID else { throw Failure.invalidReply }
    }

    private func action(_ path: String, _ id: String, _ args: [String: Any]) async throws -> [String: Any] {
        let reply = try await request(path, ["id": id, "args": args])
        guard reply["ok"] as? Bool == true, let result = reply["result"] as? [String: Any],
              result["ok"] as? Bool != false else { throw Failure.invalidReply }
        return result
    }

    private final class NoRedirects: NSObject, URLSessionTaskDelegate {
        func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
                        newRequest request: URLRequest, completionHandler: @escaping (URLRequest?) -> Void) {
            completionHandler(nil)
        }
    }
}
