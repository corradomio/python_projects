import asyncio
import asyncpg

# async def run():
#     conn = await asyncpg.connect(user='root', password='playwithdata',
#                                  database='OpenBeer',
#                                  host='127.0.0.1')
#     values = await conn.fetch(
#         'SELECT * FROM Beer LIMIT 10'
#     )
#     await conn.close()
#
# loop = asyncio.get_event_loop()
# loop.run_until_complete(run())

async def main():
    # conn = await asyncpg.connect('postgresql://postgres@localhost/test')
    conn = await asyncpg.connect(user='root', password='playwithdata', database='OpenBeer', host='127.0.0.1')
    row = await conn.fetchrow(
        'SELECT * FROM Beer LIMIT 10')
    await conn.close()



asyncio.get_event_loop().run_until_complete(main())


# def main():
#     conn = asyncpg.connect(host="172.17.229.8", port=5432, user="root", password="playwithdata")
#     conn.close()
#
#
# if __name__ == "__main__":
#     main()


