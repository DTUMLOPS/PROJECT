from google.cloud import secretmanager


def access_secret(project_id, secret_id):
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    secret_value = response.payload.data.decode("UTF-8")
    return secret_value


def get_wandb_token():
    project_id = "dtumlops-447914"
    secret_id = "WANDB_API_TOKEN"
    wandb_api_token = access_secret(project_id, secret_id)
    return wandb_api_token


if __name__ == "__main__":
    wandb_api_token = get_wandb_token()
    print(f"WandB Token: {wandb_api_token}")
