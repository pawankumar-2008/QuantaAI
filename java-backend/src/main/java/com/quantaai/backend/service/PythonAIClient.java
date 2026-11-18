package com.quantaai.backend.service;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

@Component
public class PythonAIClient {

    private final HttpClient httpClient;
    private final String baseUrl;

    public PythonAIClient(
            @Value("${quantaai.python-base-url:http://127.0.0.1:8000}")
            String baseUrl) {

        this.baseUrl = baseUrl;
        this.httpClient = HttpClient.newBuilder()
        .version(HttpClient.Version.HTTP_1_1)
        .build();
    }

    public String sendMessage(String message, String agent) {

        try {
            String jsonBody = """
                    {
                        "message": "%s",
                        "agent": "%s"
                    }
                    """.formatted(
                    escapeJson(message),
                    escapeJson(agent)
            );
                System.out.println("Outgoing JSON: [" + jsonBody + "]");
                System.out.println("Outgoing JSON bytes: " +
                jsonBody.getBytes(StandardCharsets.UTF_8).length);
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(baseUrl + "/chat"))
                    .header("Content-Type", "application/json")
                    .header("Accept", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(
                            jsonBody, StandardCharsets.UTF_8))
                    .build();

            HttpResponse<String> response = httpClient.send(
                    request,
                    HttpResponse.BodyHandlers.ofString()
            );

            System.out.println("Python HTTP Status: " + response.statusCode());
            System.out.println("Python Response: " + response.body());

            if (response.statusCode() != 200) {
                throw new RuntimeException(
                        "Python AI service error: " + response.body()
                );
            }

            String body = response.body();

            // Extract the reply field from the JSON response.
            int start = body.indexOf("\"reply\":\"");

            if (start == -1) {
                throw new RuntimeException("Reply field missing in Python response.");
            }

            start += 9;
            int end = body.indexOf("\",", start);

            if (end == -1) {
                end = body.lastIndexOf("\"");
            }

            return body.substring(start, end)
                    .replace("\\n", "\n")
                    .replace("\\\"", "\"")
                    .replace("\\\\", "\\");

        } catch (IOException | RuntimeException e) {
            throw new RuntimeException(
                    "Failed to communicate with Python AI service: "
                            + e.getMessage(), e
            );
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException(
                "Failed to communicate with Python AI service: "
                    + e.getMessage(), e
            );
        }
    }

    private String escapeJson(String value) {
        return value
                .replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t");
    }
}