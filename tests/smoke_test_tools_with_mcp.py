from app.tools import TOOLS

print("=== Loaded TOOLS ===")
for tool in TOOLS:
    print(tool["name"], "-", tool["description"])