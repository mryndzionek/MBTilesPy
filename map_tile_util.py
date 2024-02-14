from multiprocessing import Pool
from PIL import Image, ImageDraw
import sys

sys.path.insert(0, "./vector-tile-base")
import vector_tile_base
import gzip
import os
import io
import math
from functools import partial

import struct
import sqlite3
import logging


class DBMapWriter:
    def __init__(self, file_name):
        self.file_name = file_name

    def init(self):
        self.connection = sqlite3.connect(self.file_name)
        self.cursor = self.connection.cursor()

        self.cursor.execute(
            """CREATE TABLE IF NOT EXISTS tiles 
                          (zoom_level numeric, tile_column numeric,
                           tile_row numeric, tile_data blob)"""
        )

        self.cursor.execute("DELETE FROM tiles")
        self.connection.commit()

    def write(self, z, x, y, data):
        self.cursor.execute(
            """INSERT INTO tiles (zoom_level, tile_column, tile_row, tile_data)
                            VALUES (?, ?, ?, ?)""",
            (z, x, y, sqlite3.Binary(data)),
        )

    def finish(self):
        self.connection.commit()


class FSMapWriter:
    def __init__(self, dir_name, format="dat"):
        self.dir_name = dir_name
        self.format = format

    def init(self):
        os.makedirs(self.dir_name)
        os.chdir(self.dir_name)

    @staticmethod
    def __setDir(d):
        if not os.path.exists(d):
            os.makedirs(d)
        os.chdir(d)

    def write(self, z, x, y, data):
        FSMapWriter.__setDir(str(z))
        FSMapWriter.__setDir(str(x))
        output_file = open(str(y) + "." + self.format, "wb")
        output_file.write(data)
        output_file.close()
        os.chdir("..")
        os.chdir("..")

    def finish(self):
        os.chdir("..")


class MapTileUtil:

    __base_style = [
        "black",
        "maroon",
        "green",
        "olive",
        "navy",
        "purple",
        "teal",
        "silver",
        "gray",
        "red",
        "lime",
        "yellow",
        "blue",
        "fuchsia",
        "aqua",
        "white",
    ]

    __layer_style = {
        "water": "lightblue",
        "waterway": "lightblue",
        "park": "seagreen",
        "forest": "green",
        "mountain_peak": "brown",
        "landcover": "gray",
        "building": "darkgray",
        "poi": "red",
        "aeroway": "lightgray",
        "boundary": "darkblue",
        "path": "black",
        "secondary": "darkgray",
        "transit": "yellow",
        "residential": "lightgray",
        "industrial": "cadetblue",
        "cemetery": "slategrey",
        "military": "olivedrab",
        "commercial": "slateblue",
        "hospital": "slateblue",
        "motorway": "black",
        "grass": "lightgreen",
        "retail": "gray",
        "farmland": "olive",
        "allotments": "olive",
        "glacier": "white",
        "grassland": "olive",
        "meadow": "olive",
        "wood": "darkgreen",
        "orchard": "palegreen",
        "wetland": "brown",
        "railway": "midnightblue",
        "rail": "midnightblue",
        "recreation_ground": "mediumaquamarine",
        "stadium": "purple",
        "school": "navy",
        "kindergarten": "navy",
        "university": "navy",
        "college": "navy",
        "library": "navy",
        "track": "darkgray",
        "primary": "black",
        "tertiary": "black",
        "trunk": "brown",
        "transportation_name": "black",
        "minor": "black",
        "service": "black",
    }

    def __init__(self, mbtiles_file_name, tile_size=256, disp_size=(320, 480)):
        self.mbtiles_file_name = mbtiles_file_name
        self.tile_size = tile_size
        self.disp_size = disp_size
        self.lat = 0.0
        self.lon = 0.0
        self.zoom = 0

    def connect(self):
        connection = sqlite3.connect(self.mbtiles_file_name)
        self.cursor = connection.cursor()

        self.cursor.execute("SELECT value FROM metadata WHERE name='format'")
        img_format = self.cursor.fetchone()[0]
        if img_format != "pbf":
            raise ValueError("Unsupported file format: {}".format(img_format))

    def set_position(self, lat, lon):
        self.lat = lat
        self.lon = lon

    def set_zoom(self, zoom):
        self.zoom = zoom

    def set_tile_size(self, tile_size):
        self.tile_size = tile_size

    def __deg2num(self):
        lat_rad = math.radians(self.lat)
        n = 2.0**self.zoom
        xtile = (self.lon + 180.0) / 360.0 * n
        ytile = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
        xp = int(self.tile_size * (xtile % 1))
        yp = int(self.tile_size * (ytile % 1))
        return (int(xtile), int(n) - int(ytile) - 1, xp, yp)

    def get_tile_zxy(self, z, x, y):
        logging.info("Reading tile: {}-{}-{}".format(z, x, y))
        self.cursor.execute(
            "SELECT * FROM tiles WHERE zoom_level='{}' AND tile_column='{}' AND tile_row='{}'".format(
                z, x, y
            )
        )
        row = self.cursor.fetchone()
        if row:
            data = row[3]
            with gzip.GzipFile(fileobj=io.BytesIO(data)) as gz:
                data = gz.read()
                return data

    def get_tile(self):
        (x, y, _, _) = self.__deg2num()
        return self.get_tile_zxy(self.zoom, x, y)

    @staticmethod
    def __poly_winding(poly):
        return 0 > sum(
            [(x2 - x1) * (y2 - y1) for ((x1, y1), (x2, y2)) in zip(poly, poly[1:])]
        )

    @staticmethod
    def __render(img, tp, geom, style, ex, off_x, off_y, scale_x, scale_y, name=None):
        def sp(xy):
            return (int((xy[0]) * scale_x), int((xy[1]) * scale_y))

        c, w = style
        canv = ImageDraw.Draw(img)
        if tp == "polygon":
            for ps in geom:
                wb = MapTileUtil.__poly_winding(ps[0])
                tmp_img = Image.new("RGBA", img.size, (0, 0, 0, 0))
                tmp_canv = ImageDraw.Draw(tmp_img)
                for p in ps:
                    w = MapTileUtil.__poly_winding(p)
                    if w != wb:
                        tmp_canv.polygon(
                            list(map(sp, p)), fill=(0, 0, 0, 0), outline=(0, 0, 0, 0)
                        )
                    else:
                        tmp_canv.polygon(list(map(sp, p)), fill=c, outline="black")
                    # assert(not name)
                img.paste(tmp_img, (0, 0), tmp_img)
        elif tp == "line_string":
            for ln in geom:
                canv.line(list(map(sp, ln)), fill=c, width=w)
            if False:
                pts = [sp(l) for ln in geom for l in ln]
                ctr = pts[len(pts) / 2]
                logging.debug(ctr)
                canv.text(ctr, name, fill="cyan")
        elif tp == "point":
            # assert(len(geom) == 1)
            cp = sp(geom[0])
            canv.ellipse(
                (cp[0] - 1, cp[1] - 1, cp[0] + 1, cp[1] + 1), fill=c, outline=c
            )
            if False:
                canv.text((cp[0] + 5, cp[1] - 5), name, fill="cyan")

    @staticmethod
    def __get_feature_style(layer_name, attributes):
        atr = layer_name
        if layer_name == "landcover":
            atr = attributes["subclass"]
        elif layer_name in ["transportation", "landuse"]:
            atr = attributes["class"]

        w = 4 if atr == "primary" or atr == "transit" else 2

        if atr in MapTileUtil.__layer_style.keys():
            return (MapTileUtil.__layer_style[atr], w)
        else:
            return ("darkred", w)

    @staticmethod
    def render_tile(tile_size, tile):
        logging.debug("Rendering tile")
        vt = vector_tile_base.VectorTile(tile)
        ex = max([l.extent for l in vt.layers])

        scale_x = 1.0 * tile_size / ex
        scale_y = 1.0 * tile_size / ex

        img = Image.new("RGB", (tile_size, tile_size), color="khaki")

        supported_layers = [
            "boundary",
            "landcover",
            "landuse",
            "building",
            "housenumber",
            "place",
            "poi",
            "transportation",
            "transportation_name",
            "water",
            "water_name",
            "waterway",
        ]

        layer_lut = {l.name: l for l in vt.layers}

        for l in [
            layer_lut[n]
            for n in [l for l in supported_layers if l in layer_lut.keys()]
        ]:
            logging.debug("Layer name: {}".format(l.name))
            for ft in l.features:
                logging.debug("Feature type: {}".format(ft.type))
                style = MapTileUtil.__get_feature_style(l.name, ft.attributes)
                name = None
                # if "name" in ft.attributes:
                #     try:
                #         name = ft.attributes["name"]
                #     except UnicodeEncodeError:
                #         pass
                if "MAP_COLOR" in ft.attributes:
                    style = (
                        MapTileUtil.__base_style[int(ft.attributes["MAP_COLOR"])],
                        1,
                    )
                if l.name == "housenumber":
                    name = ft.attributes["housenumber"]
                MapTileUtil.__render(
                    img,
                    ft.type,
                    ft.get_geometry(),
                    style,
                    ex,
                    0,
                    0,
                    scale_x,
                    scale_y,
                    name,
                )
        return img

    def get_position_str(self):
        (x, y, _, _) = self.__deg2num()
        return "{}_{}_{}".format(self.zoom, x, y)

    def __is_visible(self, x, y):
        r1 = (0, 0, self.disp_size[0], self.disp_size[1])
        r2 = (
            x - (self.tile_size / 2),
            y - (self.tile_size / 2),
            x + (self.tile_size / 2),
            y + (self.tile_size / 2),
        )

        if (r1[0] >= r2[2]) or (r1[2] <= r2[0]) or (r1[3] <= r2[1]) or (r1[1] >= r2[3]):
            return False
        else:
            return True

    def __find_centers(self):
        (x, y, dx, dy) = self.__deg2num()
        cx = (self.disp_size[0] / 2) + (self.tile_size / 2) - dx
        cy = (self.disp_size[1] / 2) + (self.tile_size / 2) - dy
        points = set()

        def neigh(p_, xy_):
            return [
                (
                    (p_[0] + (dx * self.tile_size), p_[1] + (dy * self.tile_size)),
                    (xy_[0] + dx, xy_[1] - dy),
                )
                for dx, dy in [(-1, 0), (1, 0), (0, 1), (0, -1)]
            ]

        base = ((cx, cy), (x, y))
        to_visit = set([base])

        while len(to_visit) > 0:
            p_ = to_visit.pop()
            if self.__is_visible(*p_[0]):
                points.add(p_)
                ns = set([n for n in neigh(*p_) if n[1][0] >= 0 and n[1][1] >= 0])
                to_visit.update(ns.difference(points))

        return points

    @staticmethod
    def __draw_cross(canvas, x, y, s=12, w=4, c="black"):
        canvas.line([x - s, y, x + s, y], width=w, fill=c)
        canvas.line([x, y - s, x, y + s], width=w, fill=c)

    def render_display(self, crosshairs=False):
        logging.info("Rendering display")
        center_tile = self.get_tile()
        if center_tile:
            ps = self.__find_centers()

            disp_img = Image.new("RGB", self.disp_size, color="white")
            disp_canv = ImageDraw.Draw(disp_img)

            tiles = [
                (p[0][0], p[0][1], self.get_tile_zxy(self.zoom, p[1][0], p[1][1]))
                for p in ps
            ]
            tiles = [(x, y, t) for x, y, t in tiles if t]
            worker = partial(MapTileUtil.render_tile, self.tile_size)
            xy = [(x, y) for x, y, _ in tiles]
            tiles = [t for _, _, t in tiles]

            with Pool() as pool:
                for (x, y), tile_img in zip(xy, pool.imap(worker, tiles)):
                    disp_img.paste(
                        tile_img,
                        (int(x - (self.tile_size / 2)), int(y - (self.tile_size / 2))),
                    )
                    if crosshairs:
                        MapTileUtil.__draw_cross(disp_canv, x, y)

            if crosshairs:
                MapTileUtil.__draw_cross(
                    disp_canv, self.disp_size[0] / 2, self.disp_size[1] / 2, c="red"
                )

            return disp_img
        else:
            return None

    def get_stats(self):
        stats = {}
        num_tiles = 0
        max_raw_size = 0
        max_raw_pos = None
        layers = set()
        classes = set()
        subclasses = set()
        styles = set()
        names = []
        max_polygon_len = 0
        max_line_len = 0
        max_lines = 0
        max_polygons = 0
        max_extent = 0
        max_features = 0

        self.cursor.execute("SELECT * FROM tiles")

        for row in self.cursor:
            num_tiles += 1
            z, x, y = row[0], row[1], row[2]
            with gzip.GzipFile(fileobj=io.BytesIO(row[3])) as gz:
                raw = gz.read()
                if max_raw_size < len(raw):
                    max_raw_size = len(raw)
                    max_raw_pos = (z, x, y)
                vt = vector_tile_base.VectorTile(raw)
                layers.update([l.name for l in vt.layers])

                max_ll = 0
                max_pl = 0
                ex = max([l.extent for l in vt.layers])
                if ex > max_extent:
                    max_extent = ex

                mf = max([len(l.features) for l in vt.layers])
                if mf > max_features:
                    max_features = mf

                for l in vt.layers:
                    for ft in l.features:
                        if "subclass" in ft.attributes:
                            subclasses.add(ft.attributes["subclass"])
                        if "class" in ft.attributes:
                            classes.add(ft.attributes["class"])

                        style = MapTileUtil.__get_feature_style(l.name, ft.attributes)
                        styles.add(style)

                        if ft.type == "point" and "name" in ft.attributes:
                            try:
                                name = ft.attributes["name"].decode("ascii")
                                names.append(name)
                            except UnicodeEncodeError:
                                pass

                        if ft.type == "polygon":
                            if max_polygons < len(ft.get_geometry()):
                                max_polygons = len(ft.get_geometry())
                            try:
                                ml = max(
                                    [len(p) for ps in ft.get_geometry() for p in ps]
                                )
                            except ValueError:
                                ml = 0
                                pass
                            if ml > max_pl:
                                max_pl = ml
                        elif ft.type == "line_string":
                            if max_lines < len(ft.get_geometry()):
                                max_lines = len(ft.get_geometry())
                            ml = max([len(ln) for ln in ft.get_geometry()])
                            if ml > max_ll:
                                max_ll = ml
                if max_ll > max_line_len:
                    max_line_len = max_ll
                if max_pl > max_polygon_len:
                    max_polygon_len = max_pl

        stats["num_tiles"] = num_tiles
        stats["max_raw_tile_size"] = max_raw_size
        stats["max_raw_pos"] = max_raw_pos
        stats["layer_names"] = layers
        stats["num_layers"] = len(layers)
        stats["max_polygon_len"] = max_polygon_len
        stats["max_line_len"] = max_line_len
        stats["max_extent"] = max_extent
        # stats['classes'] = classes
        # stats['subclasses'] = subclasses
        stats["max_features"] = max_features
        stats["styles"] = styles
        stats["max_styles"] = len(styles)
        # stats['names'] = names
        stats["num_names"] = len(names)
        stats["max_lines"] = max_lines
        stats["max_polygons"] = max_polygons
        return stats

    def encode(self, writer, zipped=True):
        stats = self.get_stats()
        layer_lut = {n: i for i, n in enumerate(stats["layer_names"])}
        style_lut = {n: i for i, n in enumerate(stats["styles"])}
        type_lut = {"point": 0, "line_string": 1, "polygon": 2}
        assert len(layer_lut) <= 16
        assert len(style_lut) <= 32

        writer.init()

        self.cursor.execute("SELECT * FROM tiles")
        for row in self.cursor:
            z, x, y = row[0], row[1], row[2]
            with gzip.GzipFile(fileobj=io.BytesIO(row[3])) as gz:
                raw = gz.read()
                vt = vector_tile_base.VectorTile(raw)
                tile_data = bytes()
                for l in vt.layers:
                    for ft in l.features:
                        style = style_lut[
                            MapTileUtil.__get_feature_style(l.name, ft.attributes)
                        ]
                        layer = layer_lut[l.name]
                        tp = type_lut[ft.type]

                        geom = ft.get_geometry()

                        tile_data += struct.pack("BB", (tp << 4) | layer, style)

                        if ft.type == "point":
                            name = ""
                            # try:
                            #     name = ft.attributes['name'].decode('ascii')
                            # except KeyError, UnicodeEncodeError:
                            #     pass
                            tile_data += struct.pack("B", len(name))
                            # for a in name:
                            #     print(type(a))
                            #     tile_data += struct.pack('c', a)
                            tile_data += struct.pack("hh", geom[0][0], geom[0][1])
                        elif ft.type == "line_string":
                            count = len(geom)
                            tile_data += struct.pack("B", count)
                            for ln in geom:
                                tile_data += struct.pack("H", len(ln))
                                for p in ln:
                                    tile_data += struct.pack("hh", p[0], p[1])
                        elif ft.type == "polygon":
                            count = len(geom)
                            # todo - add internal style support
                            tile_data += struct.pack("BB", style, count)
                            for ps in geom:
                                wb = MapTileUtil.__poly_winding(ps[0])
                                num = len(ps)
                                tile_data += struct.pack("B", num)
                                for poly in ps:
                                    w = MapTileUtil.__poly_winding(poly)
                                    e = 0 if w == wb else 1
                                    tile_data += struct.pack("H", (e << 15) | len(poly))
                                    for p in poly:
                                        tile_data += struct.pack("hh", p[0], p[1])

                if zipped:
                    buf = io.BytesIO()
                    compressed = gzip.GzipFile(fileobj=buf, mode="wb")
                    compressed.write(tile_data)
                    compressed.close()
                    buf.seek(0)
                    writer.write(z, x, y, buf.read())
                else:
                    writer.write(z, x, y, tile_data)
        writer.finish()


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s",
        datefmt="%m-%d %H:%M:%S",
        filemode="w",
    )

    print(sys.argv)
    if len(sys.argv) != 4:
        logging.error("Three arguments needed: mbtile_file_name latitude longitude")
        sys.exit(1)

    loc = tuple(map(float, sys.argv[2:]))
    file_name = sys.argv[1]

    if not os.path.exists(file_name):
        logging.error("MBTiles file {} doesn't exist")
        sys.exit(1)

    mbt_util = MapTileUtil(file_name)
    mbt_util.connect()

    mbt_util.set_position(*loc)
    mbt_util.set_zoom(14)
    tile = mbt_util.get_tile()
    img = MapTileUtil.render_tile(4 * mbt_util.tile_size, tile)
    img.save("tile-{}.png".format(mbt_util.get_position_str()))

    j = 0

    for z in range(0, 15):
        mbt_util.set_zoom(z)
        img = mbt_util.render_display(crosshairs=True)
        if img:
            img.save("{:03d}.png".format(j))
            # img.save('display-{}.png'.format(mbt_util.get_position_str()))
            j = j + 1

    mbt_util.set_zoom(14)
    ts = mbt_util.tile_size
    for s in range(1, 5):
        mbt_util.set_tile_size(ts * (2**s))
        img = mbt_util.render_display(crosshairs=True)
        if img:
            img.save("{:03d}.png".format(j))
            # img.save('display-{}.png'.format(mbt_util.get_position_str()))
            j = j + 1
