#
# URL (wiki)
#   https://en.wikipedia.org/wiki/URL
# Uniform Resource Locators (URL)
#   https://datatracker.ietf.org/doc/html/rfc1738
# Uniform Resource Identifier (URI): Generic Syntax
#   https://datatracker.ietf.org/doc/html/rfc3986
#
# syntax:
#
#   A B     A and B
#   A|B     A or  B
#   (A)     grouping
#   A?      optional A
#   A*      zero or more A
#   A+      one or more A
#   [A]     optional A      (A)?
#   {A}     zero or more A  (A)*
#
#
# default:
#
#       URI = scheme ":" ["//" authority] path ["?" query] ["#" fragment]
#
#       scheme = protocol
#
#       authority = [userinfo "@"] hostport
#
#       userinfo = username ":" password
#
#       hostport = host [":" port]
#
#       query = key1 "=" value1 "&" key2 "=" value2
#               key1 "=" value1 ";" key2 "=" value2
#
# extensions:
#
#       JDBC URL:       scheme = "jdbc" ":" dialect ":" [variant ":"]
#
#           Oracle
#               jdbc:oracle:thin:[<username>/<password>]@<host>[:<port>]:<SID>
#               jdbc:oracle:thin:[<username>/<password>]@//<host>[:<port>]/<service>
#               jdbc:oracle:thin:@(DESCRIPTION=(ADDRESS=(PROTOCOL=TCP)(HOST=<host>)(PORT=<port>))(CONNECT_DATA=(SERVICE_NAME=<service>)))
#
#           MySQL
#               protocol//[hosts][/database][?properties]
#
#               protocol:
#                   jdbc:mysql:
#                   jdbc:mysql:loadbalance:
#                   jdbc:mysql:replication:
#               hosts:
#                   host1, host2,…,hostN
#               host:
#                   [username:password@]host[:port]
#               properties:
#                   prop1=value1&prop2=value2
#
#           SQLServer
#               jdbc:sqlserver://[serverName[\instanceName][:portNumber]][;property=value[;property=value]]
#
#           postgresql
#               jdbc:postgresql://host:port/database?properties
#
#           SQLite
#               jdbc:sqlite:sqlite_database_file_path
#               jdbc:sqlite:sample.db
#               jdbc:sqlite:C:/sqlite/db/chinook.db
#               jdbc:sqlite::memory:.
#
#       SQL Alchemy URL:    dialect+driver://[username:password@]host[:port]/database
#
#           postgresql
#               postgresql://scott:tiger@localhost/mydatabase
#               postgresql+psycopg2://scott:tiger@localhost:5432/mydatabase
#               postgresql+pg8000://scott:tiger@localhost/mydatabase
#
#           MySQL
#               mysql://scott:tiger@localhost/foo
#               mysql+mysqldb://scott:tiger@localhost/foo
#               mysql+pymysql://scott:tiger@localhost/foo
#
#           Oracle
#               oracle://scott:tiger@127.0.0.1:1521/sidname
#               oracle+cx_oracle://scott:tiger@tnsname
#
#           SQLServer
#               mssql+pyodbc://scott:tiger@mydsn
#               mssql+pymssql://scott:tiger@hostname:port/dbname
#
#           SQLlite
#               sqlite:////absolute/path/to/foo.db
#               sqlite:///C:\\path\\to\\foo.db
#             r"sqlite:///C:\path\to\foo.db"
#
# {
#       drivername: str,
#       username: Optional[str] = None,
#       password: Optional[str] = None,
#       host: Optional[str] = None,
#       port: Optional[int] = None,
#       database: Optional[str] = None,
#       query: Mapping[str, Union[Sequence[str], str]] = util.EMPTY_DICT,
# }
# .

import re


class URL:

    def __init__(self, url: str):
        self.parts: dict[str, str] = {'url': url}
        self.protocols: list[str]
        self.query: dict[str, str] = {}
        query = self._parse_url(url)
        self._parse_query(query)
        self._analyze_query()
    # end

    def _parse_url(self, url: str) -> str:
        # normalize the slashes: '\' -> '/'
        url = url.replace('\\', '/')

        # check if the syntax is 'd:/...'.
        # if true, convert it into 'file:///d:/...'
        if len(url) > 3 and url[1] == ':' and url[2] == '/':
            url = "file:///" + url

        # check if the url has the form 'scheme:path' -> NOT SUPPORTED
        if url.find('://') == -1:
            raise ValueError(f"Unsupported URL syntax: {url}")

        # split scheme from the rest
        scheme, rest = URL._split(url, "://")

        # parse 'scheme':
        #       jdbc:<dialect>[:<variant>]
        #       <dialect>[+<driver>]
        protocols: list[str] = re.split(r":|\+", scheme)
        if protocols[0] == 'jdbc':
            protocols = protocols[1:]
        self.parts['protocol'] = protocols[0]
        if len(protocols) > 1:
            self.parts['variant'] = protocols[1]

        # parse 'authority'
        #   [userinfo "@"] hostport
        authority, rest = URL._split(rest, "/")
        self.parts['authority'] = authority

        if '@' in authority:
            userinfo, hostport = URL._split(authority, "@")
        else:
            userinfo, hostport = "", authority

        self.parts['hostport'] = hostport

        # split 'userinfo'
        if len(userinfo) > 0:
            self.parts['userinfo'] = userinfo
            username, password = URL._split(userinfo, ":")
            self.query['username'] = username
            self.query['password'] = password

        # split hostport
        if ':' in hostport:
            host, port = URL._split(hostport, ':')
        else:
            host, port = hostport, ""

        self.parts['host'] = host
        if len(port) > 0:
            self.parts['port'] = port

        # split 'path?query#fragment'
        if '?' in rest:
            path, qf = URL._split(rest, '?')
            if '#' in qf:
                query, fragment = URL._split(rest, '#')
            else:
                query, fragment = qf, ""
        elif '#' in rest:
            query = ""
            path, fragment = URL._split(rest, '#')
        else:
            path, query, fragment = rest, "", ""

        self.parts['path'] = path
        if '/' in path:
            parent, file = URL._split(path, '/', True)
        else:
            parent, file = "", path

        if len(parent):
            self.parts['parent'] = parent

        self.parts['file'] = file

        if len(fragment) > 0:
            self.parts['fragment'] = fragment

        return query
    # end

    def _parse_query(self, query: str):
        if len(query) == 0:
            return {}
        kvdict = {}

        #
        # key1 "=" value1 "&" key2 "=" value2
        # key1 "=" value1 ";" key2 "=" value2
        #
        kvlist = re.split('&|;', query)
        for kv in kvlist:
            if "=" not in kv:
                k, v = kv, "true"
            else:
                k, v = URL._split(kv, '=')

            kvdict[k.strip()] = v.strip()
        # end
        self.query = kvdict
    # end

    def _analyze_query(self):
        if 'username' not in self.query:
            return

        username = self.query['username']
        password = self.query['password']
        self.parts['userinfo'] = username + ':' + password
    # end

    @property
    def as_sqlalchemy(self):
        url = self.parts['protocol']
        if 'variant' in self.parts:
            url += "+" + self.parts['variant']
        url += '://'
        if 'userinfo' in self.parts:
            url += self.parts['userinfo'] + '@'

        if 'hostport' in self.parts:
            url += self.parts['hostport'] + '/'

        if 'path' in self.parts:
            url += self.parts['path']

        return url

    @staticmethod
    def _split(s: str, sep: str, last: bool = False) -> tuple[str, str]:
        l = len(sep)
        pos = s.rfind(sep) if last else s.find(sep)
        if pos == -1:
            return ("", s) if last else (s, "")
        else:
            return s[0:pos], s[pos + l:]
# end
