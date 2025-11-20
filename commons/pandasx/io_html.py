import pandas as pd
import xml.etree.ElementTree as ET

#
# <table>
#     <thead>
#         <tr>
#             <th>Date  </th>
#             <th>Open  </th>
#             <th>High  </th>
#             <th>Low  </th>
#             <th>Close </th>
#             <th>Adj Close </th>
#             <th>Volume  </th>
#         </tr>
#     </thead>
#     <tbody>
#         <tr>
#             <td>Dec 31, 2013</td>
#             <td>1,842.61</td>
#             <td>1,849.44</td>
#             <td>1,842.41</td>
#             <td>1,848.36</td>
#             <td>1,848.36</td>
#             <td>2,312,840,000</td>
#         </tr>
#         <tr>
#             <td>Dec 30, 2013</td>
#             <td>1,841.47</td>
#             <td>1,842.47</td>
#             <td>1,838.77</td>
#             <td>1,841.07</td>
#             <td>1,841.07</td>
#             <td>2,293,860,000</td>
#         </tr>
#     </tbody>
# </table>

def _vparse(v: str):
    if v is None or len(v) == 0:
        return None
    v = v.strip()
    if "." in v and "," in v or "." in v:
        try:
            f = v.replace(",", "")
            return float(f)
        except:
            pass
    if "," in v:
        try:
            i = v.replace(",", "")
            return int(i)
        except:
            pass

    b = v.lower()
    if b in ["true", "on", "yes", "open"]:
        return True
    if b in ["false", "off", "no", "close"]:
        return False
    return v

def read_html_table(table_file: str) -> pd.DataFrame:
    doc = ET.parse(table_file)
    table = doc.getroot()

    # -- head --
    thead = table.find("thead")
    tr = thead.find("tr")
    columns = [td.text.strip() for td in tr.findall("th")]

    # -- data --
    data = []

    tbody = table.find("tbody")
    for tr in tbody.findall("tr"):
        # for i, td in enumerate(tr.findall("td")):
        #     data[i].append(_vparse(td.text))
        values = [_vparse(td.text) for td in tr.findall("td")]
        data.append(values)

    df = pd.DataFrame(data=data, columns=columns)
    return df
# end

