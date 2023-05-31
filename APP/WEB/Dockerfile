# Selecciona una imagen base de Python adecuada
FROM python:3.9.16

# Copia el archivo requirements.txt al contenedor
COPY requirements.txt /app/requirements.txt

# Establece el directorio de trabajo
WORKDIR /app

# Instala las dependencias del proyecto
RUN pip install -r requirements.txt

# Copia el resto de los archivos de tu proyecto al contenedor
COPY . /app

# Especifica el comando de inicio de tu aplicación
CMD ["python", "index.py"]
