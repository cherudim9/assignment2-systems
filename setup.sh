pip install uv

ssh-keygen -t ed25519

cat ~/.ssh/id_ed25519.pub 

ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts

TRITON_INTERPRETER=1  uv run pytest tests/test_attention.py -k "test_flash_forward_pass_triton"