from dotenv import load_dotenv

# NOTE: the `override=True` option is only necessary because VS Code is caching .env values,
# cf. https://stackoverflow.com/questions/77338955/vscode-does-not-update-environment-variables-upon-env-modification
load_dotenv(override=True)
