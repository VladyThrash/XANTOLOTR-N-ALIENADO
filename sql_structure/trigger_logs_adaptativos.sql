-- Función del Trigger
CREATE OR REPLACE FUNCTION trg_check_vector_similarity()
RETURNS TRIGGER AS $$
DECLARE
    last_vector vector(128);
    dist float;
BEGIN
    -- Obtener el vector más reciente del mismo usuario
    SELECT vector_caracteristicas INTO last_vector
    FROM logs
    WHERE id_usuario = NEW.id_usuario
    ORDER BY fecha_captura DESC
    LIMIT 1;

    IF FOUND THEN
        -- Calcular la distancia coseno
        dist := NEW.vector_caracteristicas <=> last_vector;
        
        -- Si es casi idéntico (distancia < 0.05), abortar el INSERT retornando NULL
        IF dist < 0.05 THEN
            RETURN NULL;
        END IF;
    END IF;

    -- Si la distancia es significativa (> 0.05), permitir el INSERT
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Enlazar la función a la tabla
CREATE TRIGGER trg_adaptive_vector
BEFORE INSERT ON logs
FOR EACH ROW
EXECUTE FUNCTION trg_check_vector_similarity();