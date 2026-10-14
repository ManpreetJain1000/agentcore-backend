-- Runs once, the first time the database volume is created
CREATE EXTENSION IF NOT EXISTS vector;

CREATE DATABASE agentcore_test;
\c agentcore_test
CREATE EXTENSION IF NOT EXISTS vector;