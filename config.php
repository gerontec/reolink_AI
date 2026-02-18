<?php
/**
 * Zentrale Datenbank-Konfiguration
 * Diese Datei wird von allen PHP-Skripten eingebunden
 */

// Datenbank-Konfiguration
define('DB_HOST', 'localhost');
define('DB_NAME', 'wagodb');
define('DB_USER', 'gh');
define('DB_PASS', 'a12345');
define('DB_CHARSET', 'utf8mb4');

/**
 * Erstellt eine PDO-Datenbankverbindung
 * @return PDO
 * @throws PDOException
 */
function getDbConnection() {
    try {
        $pdo = new PDO(
            sprintf('mysql:host=%s;dbname=%s;charset=%s', DB_HOST, DB_NAME, DB_CHARSET),
            DB_USER,
            DB_PASS,
            [
                PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION,
                PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC,
                PDO::ATTR_EMULATE_PREPARES => false
            ]
        );
        return $pdo;
    } catch (PDOException $e) {
        error_log("Database connection failed: " . $e->getMessage());
        throw $e;
    }
}