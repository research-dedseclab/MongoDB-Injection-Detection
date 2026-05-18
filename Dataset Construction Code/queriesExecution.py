import subprocess

# ============== CONFIG ==================
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "ecommerce_demo"

INPUT_FILE = "benign_queries.txt"
SUCCESS_FILE = "successful_queries.txt"
FAILED_FILE = "failed_queries.txt"
# ========================================

success_log = open(SUCCESS_FILE, "w")
failure_log = open(FAILED_FILE, "w")

with open(INPUT_FILE, "r") as file:
    for line_number, line in enumerate(file, start=1):
        query = line.strip()

        if not query:
            continue

        # Run mongosh with the query
        result = subprocess.run(
            [
                "mongosh",
                MONGO_URI + "/" + DB_NAME,
                "--quiet",
                "--eval",
                query
            ],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            success_log.write(query + "\n")
        else:
            failure_log.write(
                f"Line {line_number}: {query}\n"
                f"STDERR: {result.stderr}\n\n"
            )

success_log.close()
failure_log.close()

print("Done executing queries using mongosh.")
