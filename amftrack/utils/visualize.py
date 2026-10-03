from __future__ import annotations
import cv2


def _id_color(track_id: int):
    # Deterministic high-contrast color generated from the ID.
    r = (37 * track_id + 53) % 255
    g = (17 * track_id + 101) % 255
    b = (29 * track_id + 151) % 255
    return int(b), int(g), int(r)


def draw_tracks(frame, tracks):
    out = frame.copy()
    for track in tracks:
        x1, y1, x2, y2 = track.to_xyxy().astype(int)
        color = _id_color(track.track_id)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out,
            f"ID {track.track_id}",
            (x1, max(y1 - 7, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return out
