function export_boxes()
% Export the bounding boxes of the Svanstrom et al. IR label files
% (MATLAB groundTruth objects, tag v1.0.0) to one CSV, so that the boxes can
% be used without MATLAB. Written for this study; nothing from the dataset is
% executed. Paths come from the environment variables LABEL_DIR and OUT_CSV.
% Columns: file, row (1-based LabelData row = frame), time_s (LabelData time),
% label (label variable name), box_idx (1-based within the row and label),
% x, y, w, h (pixels, MATLAB [x y w h] convention, x/y of the top-left
% corner, 1-based image coordinates).
labelDir = getenv('LABEL_DIR');
outCsv = getenv('OUT_CSV');
v = ver('MATLAB');
fprintf('MATLAB %s %s\n', v.Version, v.Release);
tb = ver;
fprintf('toolboxes: %s\n', strjoin({tb.Name}, '; '));
files = dir(fullfile(labelDir, 'IR_*_LABELS.mat'));
fprintf('label files: %d\n', numel(files));
fid = fopen(outCsv, 'w');
fprintf(fid, 'file,row,time_s,label,box_idx,x,y,w,h\n');
nbox = 0; nrows = 0; nempty = 0;
for k = 1:numel(files)
    s = load(fullfile(labelDir, files(k).name));
    fn = fieldnames(s);
    g = s.(fn{1});
    if ~isa(g, 'groundTruth')
        error('not a groundTruth object: %s (%s)', files(k).name, class(g));
    end
    ld = g.LabelData;
    t = seconds(ld.Time);
    vars = ld.Properties.VariableNames;
    for r = 1:height(ld)
        nrows = nrows + 1;
        anyBox = false;
        for j = 1:numel(vars)
            b = ld{r, j};
            if iscell(b)
                b = b{1};
            end
            if isstruct(b)
                b = vertcat(b.Position);
            end
            for q = 1:size(b, 1)
                fprintf(fid, '%s,%d,%.6f,%s,%d,%.4f,%.4f,%.4f,%.4f\n', files(k).name, r, ...
                        t(r), vars{j}, q, b(q, 1), b(q, 2), b(q, 3), b(q, 4));
                nbox = nbox + 1;
                anyBox = true;
            end
        end
        if ~anyBox
            nempty = nempty + 1;
        end
    end
    if mod(k, 50) == 0
        fprintf('%d/%d files\n', k, numel(files));
    end
end
fclose(fid);
fprintf('rows %d, boxes %d, rows without a box %d\n', nrows, nbox, nempty);
end
