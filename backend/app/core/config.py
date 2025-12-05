from pydantic import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "FoodScan-X"
    API_V1_PREFIX: str = "/api/v1"
    DEBUG: bool = True

    # Future-ready fields
    DATABASE_URL: str = "sqlite:///./foodscan.db"   # placeholder
    SECRET_KEY: str = "super-secret-key"            # placeholder
    ABHA_API_BASE: str = "https://healthid.ndhm.gov.in"  # future integration

    class Config:
        env_file = ".env"   # allows environment overrides later

# Create a single settings instance for the app
settings = Settings()
