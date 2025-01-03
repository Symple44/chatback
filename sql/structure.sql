-- Tables du back AI CHATBOT

-- GRANT ALL PRIVILEGES ON SCHEMA public TO aioweoadmin;
-- ALTER FUNCTION update_updated_at_column OWNER TO aioweoadmin;

-- Activer les extensions nécessaires
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

-- Table des utilisateurs
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE NOT NULL,
    full_name VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP WITH TIME ZONE,
    user_metadata JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT true
);

-- Table des sessions
CREATE TABLE chat_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_id VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    session_context JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT true,
    session_metadata JSONB DEFAULT '{}'
);

-- Table de l'historique des chats
CREATE TABLE chat_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(255) REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    query TEXT NOT NULL,
    response TEXT NOT NULL,
    query_vector vector(764),
    response_vector vector(764),
    confidence_score FLOAT,
    tokens_used INTEGER,
    processing_time FLOAT,
    chat_metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Table des documents référencés
CREATE TABLE referenced_documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chat_history_id UUID REFERENCES chat_history(id) ON DELETE CASCADE,
    document_name VARCHAR(255) NOT NULL,
    page_number INTEGER,
    relevance_score FLOAT,
    content_snippet TEXT,
    document_metadata JSONB DEFAULT '{}'
);

-- Table des métriques d'utilisation
CREATE TABLE usage_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    session_id VARCHAR(255) REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
    endpoint_path VARCHAR(255) NOT NULL,
    response_time FLOAT,
    status_code INTEGER,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE api_error_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    endpoint VARCHAR(255) NOT NULL,
    error_message TEXT,
    stack_trace TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE vector_usage_stats (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    vector_type VARCHAR(50) NOT NULL,
    dimension INTEGER NOT NULL,
    usage_count INTEGER DEFAULT 0,
    avg_processing_time FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE message_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    message_id UUID REFERENCES chat_history(id) ON DELETE CASCADE,
    embedding_type VARCHAR(50) NOT NULL,
    vector float[],
    model_version VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    embedding_metadata  JSONB DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS error_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    error_type VARCHAR(100) NOT NULL,
    error_message TEXT NOT NULL,
    stack_trace TEXT,
    component VARCHAR(100),
    severity VARCHAR(20) DEFAULT 'error',
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    session_id VARCHAR(255),
    error_metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT severity_check CHECK (severity IN ('debug', 'info', 'warning', 'error', 'critical'))
);

-- Index pour optimiser les requêtes
CREATE INDEX IF NOT EXISTS idx_error_type ON error_logs(error_type);
CREATE INDEX IF NOT EXISTS idx_error_severity ON error_logs(severity);
CREATE INDEX IF NOT EXISTS idx_error_created ON error_logs(created_at DESC);

CREATE INDEX idx_message_embedding_type ON message_embeddings(embedding_type);
CREATE INDEX idx_message_vector ON message_embeddings USING gin(vector);
CREATE INDEX idx_message_metadata ON message_embeddings USING gin(embedding_metadata);

-- Index pour la recherche vectorielle
CREATE INDEX idx_chat_history_query_vector ON chat_history USING ivfflat (query_vector vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_chat_history_response_vector ON chat_history USING ivfflat (response_vector vector_cosine_ops) WITH (lists = 100);

-- Index pour optimiser les recherches
CREATE INDEX idx_chat_history_user_id ON chat_history(user_id);
CREATE INDEX idx_chat_history_session_id ON chat_history(session_id);
CREATE INDEX idx_chat_sessions_user_id ON chat_sessions(user_id);
CREATE INDEX idx_chat_history_created_at ON chat_history(created_at);
CREATE INDEX idx_referenced_documents_chat_history_id ON referenced_documents(chat_history_id);
CREATE INDEX idx_usage_metrics_user_id ON usage_metrics(user_id);
CREATE INDEX idx_usage_metrics_session_id ON usage_metrics(session_id);

-- Index pour la recherche full-text
CREATE INDEX idx_chat_history_query_text ON chat_history USING gin(to_tsvector('french', query));
CREATE INDEX idx_chat_history_response_text ON chat_history USING gin(to_tsvector('french', response));

-- Index pour les données JSON
CREATE INDEX idx_chat_sessions_context ON chat_sessions USING gin(session_context);
CREATE INDEX idx_chat_history_additional_data ON chat_history USING gin(additional_data);
CREATE INDEX idx_chat_history_tokens ON chat_history(tokens_used);
ALTER TABLE chat_history ADD CONSTRAINT chk_confidence_score 
    CHECK (confidence_score >= 0 AND confidence_score <= 1);

-- Trigger pour mettre à jour updated_at automatiquement
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_chat_sessions_updated_at
    BEFORE UPDATE ON chat_sessions
    FOR EACH ROW
    EXECUTE PROCEDURE update_updated_at_column();

-- Vue pour les statistiques utilisateur
CREATE VIEW user_chat_statistics AS
SELECT 
    u.id as user_id,
    u.username,
    COUNT(DISTINCT cs.session_id) as total_sessions,
    COUNT(ch.id) as total_messages,
    AVG(ch.confidence_score) as avg_confidence_score,
    AVG(ch.processing_time) as avg_processing_time,
    MAX(cs.updated_at) as last_activity
FROM users u
LEFT JOIN chat_sessions cs ON u.id = cs.user_id
LEFT JOIN chat_history ch ON cs.session_id = ch.session_id
GROUP BY u.id, u.username;
