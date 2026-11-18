function ProcessingIndicator({ agents, searching = false }) {
  if (agents.length === 0) return null;

  const agentNames = agents.map((agent) => agent.name).join(", ");

  return (
    <div
      className="processing-indicator"
      role="status"
      aria-live="polite"
      aria-label={
        searching
          ? `Searching the web for ${agentNames}`
          : `Waiting for responses from ${agentNames}`
      }
    >
      <div className="processing-agents" aria-hidden="true">
        {agents.map((agent) => (
          <span className="processing-agent" key={agent.id}>
            <agent.icon size={14} strokeWidth={1.8} />
            <i />
          </span>
        ))}
      </div>
      <span>
        {searching
          ? `Searching the web · ${agentNames}`
          : agents.length === 1
          ? `${agents[0].name} is working`
          : `${agents.length} agents are working`}
      </span>
      <span className="processing-dots" aria-hidden="true">
        <i />
        <i />
        <i />
      </span>
    </div>
  );
}

export default ProcessingIndicator;
