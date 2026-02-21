FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

WORKDIR /app
COPY . .

# Force update pip and install specific compatible versions
RUN pip install --upgrade pip && \
    pip install huggingface_hub==0.24.0 \
                diffusers==0.26.3 \
                transformers==4.38.2 \
                accelerate==0.27.2 \
                torch==2.2.1 \
                fastapi \
                uvicorn \
                pydantic
                
CMD ["python", "app.py", "--api", "--listen", "--port", "7860"]