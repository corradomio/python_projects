#
# URL (wiki)
#   https://en.wikipedia.org/wiki/URL
# Uniform Resource Locators (URL)
#   https://datatracker.ietf.org/doc/html/rfc1738
# Uniform Resource Identifier (URI): Generic Syntax
#   https://datatracker.ietf.org/doc/html/rfc3986
#
# default:
#
#       URI = scheme ":" ["//" authority] path ["?" query] ["#" fragment]
#
#       scheme = protocol
#
#       authority = [userinfo "@"] hostport
#
#       userinfo = user ":" password
#
#       hostport = host [":" port]
#
#       query = key1 "=" value1 "&" key2 "=" value2
#               key1 "=" value1 ";" key2 "=" value2
#
# extensions
#
#       scheme = (protocol "+")* (protocol ":")+
#              | letter ":"             (as file:///<disk>:/...
#
#       <single_char>:<path>...
#
import re


class URL:

    def __init__(self, url: str):
        self.protocols: list[str]
        self.query: dict[str, str] = {}
        self.parts: dict[str, str] = {'url': url}
        query = self._parse_url(url)
        self._parse_query(query)
    # end

    def _parse_url(self, url: str) -> str:
        #
        schema, rest = URL._split(url, "://")

        # parse 'schema'
        protocols: list[str] = re.split(r":|\+", schema)
        self.protocols = protocols
        if len(protocols) > 1 and protocols[0] == "jdbc":
            self.parts['protocol'] = protocols[1]
        else:
            self.parts['protocol'] = protocols[0]

        # parse 'authority'
        # authority = [userinfo "@"] host [":" port]
        authority, rest = URL._split(rest, "/")
        self.parts['authority'] = authority
        if '@' in authority:
            userinfo, hostport = URL._split(authority, "@")
        else:
            userinfo, hostport = "", authority

        self.parts['userinfo'] = userinfo
        self.parts['hostport'] = hostport

        # split userinfo
        if len(userinfo) > 0:
            user, password = URL._split(userinfo, ":")
            self.parts['user'] = user
            self.parts['password'] = password

        # split hostport
        if ':' in hostport:
            host, port = URL._split(hostport, ':')
        else:
            host, port = hostport, "0"

        self.parts['host'] = host
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
        self.parts['parent'] = parent
        self.parts['file'] = file

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

    @staticmethod
    def _split(s: str, sep: str, last: bool = False) -> tuple[str, str]:
        l = len(sep)
        pos = s.rfind(sep) if last else pos = s.find(sep)
        if pos == -1:
            return ("", s) if last else (s, "")
        else:
            return s[0:pos], s[pos + l:]
# end
