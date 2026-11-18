package com.quantaai.backend;

import java.sql.Connection;
import java.util.HashMap;
import java.util.Map;
import javax.sql.DataSource;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class DatabaseController {

    private final DataSource dataSource;

    public DatabaseController(DataSource dataSource) {
        this.dataSource = dataSource;
    }

    @GetMapping("/api/database/health")
    public Map<String, String> databaseHealth() {
        Map<String, String> result = new HashMap<>();

        try (Connection connection = dataSource.getConnection()) {
            result.put("database", "Neon PostgreSQL");
            result.put("status", "CONNECTED");
            result.put("catalog", connection.getCatalog());
        } catch (Exception e) {
            result.put("database", "Neon PostgreSQL");
            result.put("status", "FAILED");
            result.put("error", e.getMessage());
        }

        return result;
    }
}