from pydantic import BaseModel
from typing import List

class ArtistProfileRequest(BaseModel):

    artistId: int

    artistName: str

    imageUrls: List[str]


class ArtistProfileResponse(BaseModel):

    styleDesc: str

    colors: List[str]

    tags: List[str]

    profileImg: str