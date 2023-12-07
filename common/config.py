import dotenv
import os

dotenv.load_dotenv()

try:
    BRANCH = os.environ["BRANCH"]
    PREFIX = f"/hack4cure/{BRANCH}/api"
except KeyError:
    BRANCH = None
    PREFIX = ""

if BRANCH is None:
    GOOGLE_REDIRECT_URI = "http://localhost:7000/v1/auth/google/callback"
else:
    GOOGLE_REDIRECT_URI = f"https://dene.sh{PREFIX}/v1/auth/google/callback"

from tortoise import Tortoise

# Get database details from environment variables
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')
db_host = os.getenv('DB_HOST')
db_port = os.getenv('DB_PORT')
db_name = os.getenv('DB_NAME')

db_url = f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'

Tortoise.init(
    db_url=db_url
)