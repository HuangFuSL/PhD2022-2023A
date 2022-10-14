import subprocess
import datetime

if __name__ == "__main__":
    # Get the current date and time
    now = datetime.datetime.now()
    string = now.strftime("%Y-%m-%d %H:%M:%S")
    
    current_branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip()
    print(current_branch)

    # Commit the changes
    subprocess.call(['git', 'add', '.'])
    subprocess.call(['git', 'commit', '-m', string])
    subprocess.call(['git', 'push', 'origin', current_branch])

    # Pull and merge
    subprocess.call(['git', 'pull', 'origin', 'main'])
    subprocess.call(['git', 'merge', 'origin', 'main'])
    subprocess.call(['git', 'push', 'origin', 'main'])
