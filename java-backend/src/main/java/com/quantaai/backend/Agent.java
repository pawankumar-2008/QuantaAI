package com.quantaai.backend;

public class Agent {

private final Long agentId;
private final String agentKey;
private final String agentName;
private final String description;
private final boolean enabled;

public Agent(Long agentId, String agentKey, String agentName,
             String description, boolean enabled) {
    this.agentId = agentId;
    this.agentKey = agentKey;
    this.agentName = agentName;
    this.description = description;
    this.enabled = enabled;
}

public Long getAgentId() {
    return agentId;
}

public String getAgentKey() {
    return agentKey;
}

public String getAgentName() {
    return agentName;
}

public String getDescription() {
    return description;
}

public boolean isEnabled() {
    return enabled;
}

}
