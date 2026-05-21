-- DentaScan — Inicialización de base de datos
-- Compatible con SQLite y PostgreSQL
-- Para SQLite, SQLAlchemy crea las tablas automáticamente en startup.
-- Para PostgreSQL, ejecuta este script manualmente o usa Alembic.

-- Tabla de usuarios
CREATE TABLE IF NOT EXISTS users (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    email       VARCHAR(255) UNIQUE NOT NULL,
    full_name   VARCHAR(255) NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    role        VARCHAR(20) NOT NULL DEFAULT 'student'
                CHECK (role IN ('admin', 'odontologist', 'student')),
    is_active   BOOLEAN NOT NULL DEFAULT TRUE,
    created_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at  DATETIME
);

CREATE INDEX IF NOT EXISTS ix_users_email ON users(email);

-- Tabla de análisis
CREATE TABLE IF NOT EXISTS analyses (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id             INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    original_filename   VARCHAR(500) NOT NULL,
    stored_filename     VARCHAR(500) NOT NULL,
    file_format         VARCHAR(20),
    xray_type           VARCHAR(20) DEFAULT 'unknown',
    status              VARCHAR(20) NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending','processing','completed','failed')),
    error_message       TEXT,

    -- Rutas de salida
    output_preprocessed VARCHAR(500),
    output_edges        VARCHAR(500),
    output_mask         VARCHAR(500),
    output_annotated    VARCHAR(500),
    output_histogram    VARCHAR(500),

    -- Resultados ML
    features            TEXT,       -- JSON serialized
    predicted_class     INTEGER,
    predicted_label     VARCHAR(100),
    confidence_score    REAL,
    class_probabilities TEXT,       -- JSON serialized
    model_used          VARCHAR(50),
    processing_time_ms  REAL,
    dicom_metadata      TEXT,       -- JSON serialized

    created_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at  DATETIME
);

CREATE INDEX IF NOT EXISTS ix_analyses_user_id   ON analyses(user_id);
CREATE INDEX IF NOT EXISTS ix_analyses_status    ON analyses(status);
CREATE INDEX IF NOT EXISTS ix_analyses_created   ON analyses(created_at);
