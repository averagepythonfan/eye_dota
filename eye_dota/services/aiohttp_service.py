import aiohttp
from .misc import query


class AiohttpService:

    def __init__(self) -> None:
        pass


    async def get_matches_update_request(self, last_date, end_date):
        async with aiohttp.ClientSession() as s:
            async with s.get(
                "https://api.opendota.com/api/explorer",
                params={"sql": query.format(start_date=last_date, end_date=end_date, limit=3000)}
            ) as resp:
                if resp.status == 200:
                    json = await resp.json()
                    if len(json["rows"]) != 0:
                        return json["rows"]
