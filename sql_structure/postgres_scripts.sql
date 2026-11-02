-- Habilitar la extensión pgvector (Obligatorio ejecutar esto primero)
CREATE EXTENSION IF NOT EXISTS vector;

-- Tabla Catálogo (Identidades)
CREATE TABLE person (
    id_usuario BIGSERIAL PRIMARY KEY,
    nombre_completo VARCHAR(150) NOT NULL,
    fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    activo BOOLEAN DEFAULT TRUE
);

-- Tabla Transaccional (Descriptores)
CREATE TABLE logs (
    id_descriptor BIGSERIAL PRIMARY KEY,
    id_usuario BIGINT NOT NULL,
    vector_caracteristicas vector(128) NOT NULL,
    fecha_captura TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_person 
        FOREIGN KEY (id_usuario) 
        REFERENCES person(id_usuario) 
        ON DELETE CASCADE
);

-- Índice de búsqueda vectorial (Crucial para el rendimiento)
-- HNSW (Hierarchical Navigable Small World) es el algoritmo más eficiente.
-- Usamos 'vector_cosine_ops' porque OpenCV SFace se evalúa mediante Distancia Coseno.
CREATE INDEX idx_vector_cosine 
ON logs 
USING hnsw (vector_caracteristicas vector_cosine_ops);