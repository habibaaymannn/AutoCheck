from config.ConfigManager import ConfigManager

cm = ConfigManager()
cm.parse("config.yaml")
cm.validate()

print("Config loaded successfully!")
print("Mode:", cm.mode)
print("Configs:", cm.configs)