from config.ConfigManager import ConfigManager, ConfigParseError, ConfigValidationError
import sys

if __name__ == "__main__":
    cm = ConfigManager()
    try:
        cm.parse("config.yaml")
        cm.validate()
        print("Config loaded successfully!")
    except (ConfigParseError, ConfigValidationError) as e:
        msg = str(e).replace("\n", " | ")
        print(f"❌[CONFIG ERROR] Unexpected configuration failure.\n - {msg}")
        sys.exit(1)
    except Exception as e:
        print(f"❌[CONFIG ERROR] Unexpected configuration failure.\n - {e}")
        sys.exit(1)
