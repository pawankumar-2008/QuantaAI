package com.quantaai.backend.orchestrator;

import org.springframework.stereotype.Service;

import com.quantaai.backend.service.StudyAgentService;

@Service
public class AgentOrchestrator {

    private final StudyAgentService studyAgent;

    public AgentOrchestrator(StudyAgentService studyAgent) {
        this.studyAgent = studyAgent;
    }

    public String process(String agent, String message) {

        if (agent == null || agent.isBlank()) {
            agent = "Study";
        }

        if (agent.equalsIgnoreCase("Study")) {
            return studyAgent.process(message);
        }

        throw new IllegalArgumentException(
            "Agent not implemented: " + agent
        );
    }
}