from typing import Optional


def parse_url(url: str, username: Optional[str] = None, password: Optional[str] = None) -> dict:
    # dialect+driver://username:password@host:port/database
    pos = url.find("://")
    drivername = url[0:pos]
    url = url[pos+3:]

    pos = url.find("@")
    if pos != -1:
        user_password = url[:pos]
        url = url[pos+1:]
    else:
        user_password = None
    pos = url.find('/')
    host_port = url[:pos]
    database = url[pos+1:]

    if user_password is not None:
        pos = user_password.find(':')
        if pos != -1:
            username = user_password[:pos]
            password = user_password[pos+1]
        else:
            username = user_password
            password = None
    # end

    pos = host_port.find(':')
    if pos != -1:
        host = host_port[:pos]
        port = int(host_port[pos+1:])
    else:
        host = host_port
        port = 0
    # end

    dbinfo = {
        "drivername": drivername,
        "host": host,
        "database": database
    }
    if port > 0:
        dbinfo["port"] = port
    if username is not None:
        dbinfo["username"] = username
    if password is not None:
        dbinfo["password"] = password
    return dbinfo





