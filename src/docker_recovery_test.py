import subprocess


def test_docker_recovery():
    print("\nTest 4: Docker Container Recovery")

    container_name = "loan-recovery-test"

    try:
        # Run a temporary container that exits with an error
        result = subprocess.run(
            [
                "docker", "run",
                "--name", container_name,
                "alpine",
                "sh", "-c", "exit 1"
            ],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print("Failure detected: Container exited with an error.")

        # Remove the failed container
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True,
            text=True
        )

        # Verify Docker can run a new container
        recovery = subprocess.run(
            ["docker", "run", "--rm", "alpine", "echo", "Recovery successful"],
            capture_output=True,
            text=True
        )

        if recovery.returncode == 0:
            print("Recovery successful!")
            print(recovery.stdout.strip())
            return True

        print("Recovery failed.")
        return False

    except FileNotFoundError:
        print("Docker command not found.")
        return False


if __name__ == "__main__":
    test_docker_recovery()