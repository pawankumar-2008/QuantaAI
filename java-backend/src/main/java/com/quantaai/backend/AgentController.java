package com.quantaai.backend;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.util.ArrayList;
import java.util.List;

import javax.sql.DataSource;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/agents")
public class AgentController {

    private final DataSource dataSource;

    public AgentController(DataSource dataSource) {
        this.dataSource = dataSource;
    }

    @GetMapping
    public List<Agent> getAgents() throws Exception {

        List<Agent> agents = new ArrayList<>();

        String sql = """
            SELECT agent_id, agent_key, agent_name,
                   description, is_enabled
            FROM agents
            ORDER BY agent_id
            """;

        try (Connection connection = dataSource.getConnection();
             PreparedStatement statement = connection.prepareStatement(sql);
             ResultSet resultSet = statement.executeQuery()) {

            while (resultSet.next()) {

                Agent agent = new Agent(
                    resultSet.getLong("agent_id"),
                    resultSet.getString("agent_key"),
                    resultSet.getString("agent_name"),
                    resultSet.getString("description"),
                    resultSet.getBoolean("is_enabled")
                );

                agents.add(agent);
            }
        }

        return agents;
    }
}