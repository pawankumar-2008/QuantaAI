package com.quantaai.backend.service;

import com.quantaai.backend.orchestrator.AIAgent;
import org.springframework.stereotype.Service;

@Service
public class StudyAgentService implements AIAgent {

    private final PythonAIClient pythonAIClient;

    public StudyAgentService(PythonAIClient pythonAIClient) {
        this.pythonAIClient = pythonAIClient;
    }

    @Override
    public String getAgentName() {
        return "Study";
    }

    @Override
    public String process(String message) {
        return pythonAIClient.sendMessage(message, "Study");
    }
}