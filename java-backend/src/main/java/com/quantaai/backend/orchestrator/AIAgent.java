package com.quantaai.backend.orchestrator;

public interface AIAgent {

    String getAgentName();

    String process(String message);
}