import platform
import os

class OSAdapter:
    def __init__(self):
        self.os_name = platform.system().lower()
        self.os_version = platform.release()
        self.shell_type = os.environ.get("SHELL", "")

    def get_os_details(self):
        return {
            "name": self.os_name,
            "version": self.os_version,
            "shell": self.shell_type
        }

    def is_linux(self):
        return "linux" in self.os_name

    def is_macos(self):
        return "darwin" in self.os_name

    def is_windows(self):
        return "windows" in self.os_name

    def get_package_manager(self):
        if self.is_linux():
            if os.path.exists("/etc/debian_version"):
                return "apt"
            elif os.path.exists("/etc/redhat-release"):
                return "yum"
            else:
                return "unknown_linux_pm"
        elif self.is_macos():
            return "brew"
        elif self.is_windows():
            return "choco"
        return "unknown"

    def get_command_template(self, action, package_name):
        pm = self.get_package_manager()
        if action == "install":
            if pm == "apt":
                return f"sudo apt update && sudo apt install -y {package_name}"
            elif pm == "yum":
                return f"sudo yum install -y {package_name}"
            elif pm == "brew":
                return f"brew install {package_name}"
            elif pm == "choco":
                return f"choco install {package_name} -y"
        elif action == "uninstall":
            if pm == "apt":
                return f"sudo apt remove -y {package_name}"
            elif pm == "yum":
                return f"sudo yum remove -y {package_name}"
            elif pm == "brew":
                return f"brew uninstall {package_name}"
            elif pm == "choco":
                return f"choco uninstall {package_name} -y"
        elif action == "update":
            if pm == "apt":
                return f"sudo apt update && sudo apt upgrade -y {package_name}"
            elif pm == "yum":
                return f"sudo yum update -y {package_name}"
            elif pm == "brew":
                return f"brew upgrade {package_name}"
            elif pm == "choco":
                return f"choco upgrade {package_name} -y"
        elif action == "update_all":
            if pm == "apt":
                return "sudo apt update && sudo apt upgrade -y && sudo apt autoremove -y"
            elif pm == "yum":
                return "sudo yum update -y"
            elif pm == "brew":
                return "brew update && brew upgrade"
            elif pm == "choco":
                return "choco upgrade all -y"