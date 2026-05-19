import uuid
import aiofiles

from fastapi import UploadFile
from app.core.config import UPLOAD_DIR


async def save_upload_file(file: UploadFile):

    file_extension = file.filename.split(".")[-1]
    filename = (f"{uuid.uuid4()}.{file_extension}")
    file_path = UPLOAD_DIR / filename

    async with aiofiles.open(file_path, "wb") as out_file:

        content = await file.read()

        await out_file.write(content)

    return file_path