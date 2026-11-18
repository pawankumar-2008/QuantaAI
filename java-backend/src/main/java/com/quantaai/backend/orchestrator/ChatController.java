package com.quantaai.backend.orchestrator;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;

@RestController
@RequestMapping("/api/chat")
public class ChatController {

    private final AgentOrchestrator orchestrator;

    public ChatController(AgentOrchestrator orchestrator) {
        this.orchestrator = orchestrator;
    }

    public record ChatRequest(
        @NotBlank String message,
        String agent
    ) {}

    public record ChatResponse(
        String reply,
        String agent,
        String status
    ) {}

    @PostMapping
    public ResponseEntity<ChatResponse> chat(
            @Valid @RequestBody ChatRequest request) {

        String agent = request.agent() == null
                ? "Study"
                : request.agent();

        String reply = orchestrator.process(
                agent,
                request.message()
        );

        return ResponseEntity.ok(
                new ChatResponse(reply, agent, "success")
        );
    }
}