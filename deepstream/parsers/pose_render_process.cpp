#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>
#include <algorithm>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

#include "nvcustomprocess.h"

#include <gst/gst.h>
#include "nvdsmeta.h"
#include "gstnvdsinfer.h"
#include "realsense_meta.h"
#include "render_meta.h"

static void write_nv12(const cv::Mat &bgr, NvDsCustomPostProcessFrameInCpu *in) {
    const int W = in->W, H = in->H;
    cv::Mat i420; cv::cvtColor(bgr, i420, cv::COLOR_BGR2YUV_I420);
    const uint8_t *ysrc = i420.data;
    const uint8_t *usrc = ysrc + (size_t)W * H;
    const uint8_t *vsrc = usrc + (size_t)(W / 2) * (H / 2);
    for (int r = 0; r < H; ++r)
        memcpy(in->Y_host + (size_t)r * in->Y_pitch, ysrc + (size_t)r * W, (size_t)W);
    const int cW = W / 2, cH = H / 2;
    for (int r = 0; r < cH; ++r) {
        uint8_t *dst = in->UV_host + (size_t)r * in->UV_pitch;
        const uint8_t *u = usrc + (size_t)r * cW, *v = vsrc + (size_t)r * cW;
        for (int c = 0; c < cW; ++c) { dst[2*c] = u[c]; dst[2*c+1] = v[c]; }
    }
}

static cv::Mat read_nv12_bgr(NvDsCustomPostProcessFrameInCpu *in) {
    const int W = in->W, H = in->H;
    cv::Mat yuv(H + H / 2, W, CV_8UC1);
    for (int r = 0; r < H; ++r)
        memcpy(yuv.ptr(r), in->Y_host + (size_t)r * in->Y_pitch, (size_t)W);
    for (int r = 0; r < H / 2; ++r)
        memcpy(yuv.ptr(H + r), in->UV_host + (size_t)r * in->UV_pitch, (size_t)W);
    cv::Mat bgr; cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_NV12); return bgr;
}

static const RsMeta *find_rs_meta(NvDsFrameMeta *fm) {
    if (!fm) return nullptr;
    static gsize t = 0;
    if (g_once_init_enter(&t)) {
        gsize v = (gsize) nvds_get_user_meta_type((gchar *) RS_NVDS_META_NAME);
        g_once_init_leave(&t, v);
    }
    for (NvDsMetaList *l = fm->frame_user_meta_list; l; l = l->next) {
        NvDsUserMeta *um = (NvDsUserMeta *) l->data;
        if (um && um->base_meta.meta_type == (NvDsMetaType) t)
            return (const RsMeta *) um->user_meta_data;
    }
    return nullptr;
}

struct TensorRef { const float *data; int d[4]; int ndim; long total; const char *name; };

static std::vector<TensorRef> collect_tensors(NvDsCustomPostProcessFrameInCpu *in) {
    std::vector<TensorRef> out;
    for (int mi = 0; mi < (int) in->num_tensor_metas; ++mi) {
        NvDsInferTensorMeta *tm = (NvDsInferTensorMeta *) in->tensor_metas[mi];
        if (!tm) continue;
        for (unsigned li = 0; li < tm->num_output_layers; ++li) {
            const float *host = (const float *) tm->out_buf_ptrs_host[li];
            if (!host) continue;
            NvDsInferLayerInfo &info = tm->output_layers_info[li];
            TensorRef t; t.data = host; t.name = info.layerName;
            int nd = (int) info.inferDims.numDims, base = 0;
            if (nd > 0 && info.inferDims.d[0] == 1) { base = 1; nd--; }
            t.ndim = nd > 4 ? 4 : nd; t.total = 1;
            for (int i = 0; i < 4; ++i) t.d[i] = 1;
            for (int i = 0; i < t.ndim; ++i) { t.d[i] = info.inferDims.d[base + i]; t.total *= t.d[i]; }
            out.push_back(t);
        }
    }
    return out;
}

static const TensorRef *find_tensor(const std::vector<TensorRef> &ts,
                                    int a, int b = -1, int c = -1, int d = -1) {
    int want = (b < 0) ? 1 : (c < 0) ? 2 : (d < 0) ? 3 : 4;
    for (const TensorRef &t : ts) {
        if (t.ndim != want || t.d[0] != a) continue;
        if (b >= 0 && t.d[1] != b) continue;
        if (c >= 0 && t.d[2] != c) continue;
        if (d >= 0 && t.d[3] != d) continue;
        return &t;
    }
    return nullptr;
}

static std::vector<int> nms_classaware(const std::vector<cv::Rect> &boxes,
        const std::vector<float> &scores, const std::vector<int> &classIds,
        float scoreThr, float iouThr) {
    std::vector<cv::Rect> off(boxes.size());
    const float STRIDE = 100000.0f;
    for (size_t i = 0; i < boxes.size(); ++i) {
        int s = (int)(classIds[i] * STRIDE);
        off[i] = cv::Rect(boxes[i].x + s, boxes[i].y + s, boxes[i].width, boxes[i].height);
    }
    std::vector<int> keep;
    cv::dnn::NMSBoxes(off, scores, scoreThr, iouThr, keep);
    return keep;
}

static void place_panel(cv::Mat &canvas, const cv::Mat &panel,
                        int r, int c, int rows, int cols) {
    int cw = canvas.cols / cols, ch = canvas.rows / rows;
    if (cw <= 0 || ch <= 0 || panel.empty()) return;
    cv::Mat dst = canvas(cv::Rect(c * cw, r * ch, cw, ch));
    cv::Mat resized; cv::resize(panel, resized, dst.size());
    resized.copyTo(dst);
}

static cv::Mat colormap_minmax(const float *data, int h, int w, int cvColormap) {
    cv::Mat f(h, w, CV_32FC1, (void *) data);
    double mn = 0, mx = 0; cv::minMaxLoc(f, &mn, &mx);
    double scale = (mx > mn) ? 255.0 / (mx - mn) : 0.0;
    cv::Mat u8; f.convertTo(u8, CV_8UC1, scale, -mn * scale);
    cv::Mat color; cv::applyColorMap(u8, color, cvColormap);
    return color;
}

// ── COCO-17 skeleton topology (from PIR pseudo_code) ────────────────────────
static const int SKELETON[][2] = {
    {0,1},{0,2},{1,3},{2,4},
    {5,6},{5,7},{7,9},{6,8},{8,10},
    {5,11},{6,12},{11,12},{11,13},{13,15},{12,14},{14,16}
};
static constexpr int NKPT  = 17;
static constexpr int NLIMB = (int)(sizeof(SKELETON) / sizeof(SKELETON[0]));
static constexpr int NET   = 640;
static constexpr float SCORE_THR = 0.25f;
static constexpr float NMS_IOU   = 0.45f;
static constexpr float KPT_THR   = 0.30f;
static constexpr int NCOLS = 56;
static constexpr int NANCHORS = 8400;

extern "C" bool NvDsCustomPostProcess(NvDsCustomPostProcessFrameInCpu *in) {
    if (!in || !in->Y_host || !in->UV_host) return false;
    const int W = in->W, H = in->H;
    if (W < 2 || H < 2 || (W & 1) || (H & 1)) {
        fprintf(stderr, "[pose_render] surface must be even-sized, got %dx%d\n", W, H);
        return false;
    }

    NvDsInferTensorMeta *tm = in->num_tensor_metas
        ? (NvDsInferTensorMeta *) in->tensor_metas[0] : nullptr;
    const RsMeta *rs = find_rs_meta((NvDsFrameMeta *) in->frame_meta);
    (void) tm; (void) rs;

    // ==== BEGIN GENERATED PROCESS ====

    // Find the pose tensor [56, 8400] from the pose model (gie_id=2).
    std::vector<TensorRef> ts = collect_tensors(in);
    const TensorRef *pose_t = find_tensor(ts, NCOLS, NANCHORS);
    if (!pose_t) {
        // No tensor yet — just pass through the frame.
        return true;
    }
    const float *raw = pose_t->data; // shape [56, 8400], row-major

    // Compute letterbox inverse transform at runtime.
    const float fw = (float)W;
    const float fh = (float)H;
    const float scale = std::min<float>((float)NET / fw, (float)NET / fh);
    const int nw = (int)std::round(fw * scale);
    const int nh = (int)std::round(fh * scale);
    const float pad_x = (float)(NET - nw) / 2.0f;
    const float pad_y = (float)(NET - nh) / 2.0f;

    // Decode: tensor is [56, 8400]. Row c, column i => raw[c * NANCHORS + i].
    // Rows 0-3: cx, cy, w, h (in network coords)
    // Row 4: person confidence
    // Rows 5..55: 17 keypoints × 3 (kx, ky, kconf)

    // First pass: threshold on person confidence
    struct Det {
        float x1, y1, x2, y2, score;
        float kx[NKPT], ky[NKPT], kc[NKPT];
    };

    std::vector<Det> dets;
    dets.reserve(256);

    for (int i = 0; i < NANCHORS; ++i) {
        float conf = raw[4 * NANCHORS + i];
        if (conf < SCORE_THR) continue;

        float cx = raw[0 * NANCHORS + i];
        float cy = raw[1 * NANCHORS + i];
        float bw = raw[2 * NANCHORS + i];
        float bh = raw[3 * NANCHORS + i];

        // Map box to frame coords
        float fcx = (cx - pad_x) / scale;
        float fcy = (cy - pad_y) / scale;
        float fbw = bw / scale;
        float fbh = bh / scale;

        Det d;
        d.x1 = fcx - fbw * 0.5f;
        d.y1 = fcy - fbh * 0.5f;
        d.x2 = fcx + fbw * 0.5f;
        d.y2 = fcy + fbh * 0.5f;
        d.score = conf;

        for (int k = 0; k < NKPT; ++k) {
            int base_row = 5 + k * 3;
            float kx_net = raw[(base_row + 0) * NANCHORS + i];
            float ky_net = raw[(base_row + 1) * NANCHORS + i];
            float kc_val = raw[(base_row + 2) * NANCHORS + i];
            d.kx[k] = (kx_net - pad_x) / scale;
            d.ky[k] = (ky_net - pad_y) / scale;
            d.kc[k] = kc_val;
        }
        dets.push_back(d);
    }

    // NMS (single class = person)
    std::vector<cv::Rect> boxes(dets.size());
    std::vector<float> scores(dets.size());
    std::vector<int> classIds(dets.size(), 0);
    for (size_t i = 0; i < dets.size(); ++i) {
        const Det &d = dets[i];
        int bx = (int)std::floor(d.x1);
        int by = (int)std::floor(d.y1);
        int bw2 = (int)std::ceil(d.x2 - d.x1);
        int bh2 = (int)std::ceil(d.y2 - d.y1);
        boxes[i] = cv::Rect(bx, by, bw2, bh2);
        scores[i] = d.score;
    }
    std::vector<int> keep;
    if (!boxes.empty()) {
        cv::dnn::NMSBoxes(boxes, scores, SCORE_THR, NMS_IOU, keep);
    }

    // Build final list of persons after NMS
    struct Person {
        float kx[NKPT], ky[NKPT], kc[NKPT];
        float score;
    };
    std::vector<Person> people;
    people.reserve(keep.size());
    for (int idx : keep) {
        Person p;
        p.score = dets[idx].score;
        for (int k = 0; k < NKPT; ++k) {
            p.kx[k] = dets[idx].kx[k];
            p.ky[k] = dets[idx].ky[k];
            p.kc[k] = dets[idx].kc[k];
        }
        people.push_back(p);
    }

    // Draw skeletons and keypoints on the frame
    cv::Mat frame = read_nv12_bgr(in);

    render_meta::Builder sb(W, H);

    for (size_t pi = 0; pi < people.size(); ++pi) {
        const Person &p = people[pi];

        render_meta::Skeleton sk(NKPT);

        for (int k = 0; k < NKPT; ++k) {
            if (p.kc[k] >= KPT_THR) {
                sk.set((size_t)k, (double)p.kx[k], (double)p.ky[k]);
                cv::circle(frame,
                           cv::Point((int)std::round(p.kx[k]), (int)std::round(p.ky[k])),
                           4, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
            }
        }

        for (int li = 0; li < NLIMB; ++li) {
            int a = SKELETON[li][0];
            int b = SKELETON[li][1];
            sk.link((size_t)a, (size_t)b);
            if (p.kc[a] >= KPT_THR && p.kc[b] >= KPT_THR) {
                cv::line(frame,
                         cv::Point((int)std::round(p.kx[a]), (int)std::round(p.ky[a])),
                         cv::Point((int)std::round(p.kx[b]), (int)std::round(p.ky[b])),
                         cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
            }
        }

        sb.skeleton(sk, "person", -1, (double)p.score);
    }

    write_nv12(frame, in);
    sb.publish(in->frame_meta);
    // ==== END GENERATED PROCESS ====
    return true;
}

CHECK_NVCUSTOMPROCESS_CPU_PROTOTYPE(NvDsCustomPostProcess);