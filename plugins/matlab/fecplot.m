function f = fecplot(dim, x, y, z, c, node_map, varargin)
% Filled patch contour for unstructured 2D/3D surface zones
%
% Call:
%     h = feplot(dim, x, y, z, c, node_map, ...)
%
% Args:
%     dim: 2 or 3
%     x,y[,z]: nodal coordinate column vectors (Nx1)
%     c: contour values; either nodal (Nx1) or cell-centered (Mcells x 1)
%     node_map: M x K integer matrix of node indices per cell (K=3 or 4)
%     varargin: name-value pairs forwarded to patch (e.g. 'EdgeColor','none')
%
% Returns:
%     f: handle to the patch object
%
% Notes:
%  - If c is nodal, uses 'FaceVertexCData' with 'FaceColor'='interp'
%  - If c is cell-centered, uses 'CData' per-face with 'FaceColor'='flat'
%  - For dim==3, z must be provided and plotting uses 3D patch (viewable with lighting)
%  - node_map may contain NaNs for unequal-sided cells; those faces are handled

% Validate dim
if ~ismember(dim, [2,3])
    error('dim must be 2 or 3');
end

% Ensure column vectors
x = x(:); y = y(:);
if dim == 3
    if nargin < 4 || isempty(z)
        error('z must be provided for dim==3');
    end
    z = z(:);
end

% Determine number of nodes and faces
nNodes = numel(x);
[nFaces, ~] = size(node_map);

% Validate node_map indices
if any(node_map(:) > nNodes) || any(node_map(:) < 0 & ~isnan(node_map(:)))
    error('node_map contains invalid node indices.');
end

% Prepare vertices matrix
if dim == 2
    verts = [x, y];
else
    verts = [x, y, z];
end

% Determine if c is nodal or cell-centered
c = c(:);
isNodal = (numel(c) == nNodes);
isCell   = (numel(c) == nFaces);
if ~(isNodal || isCell)
    error('Length of c must match number of nodes (nodal) or number of faces (cell-centered).');
end

% Clean node_map: replace zeros -> NaN (if any), ensure double
node_map = double(node_map);
node_map(node_map==0) = NaN;

% If node_map contains NaNs for some cells, patch accepts NaN-separated faces.
% Create faces in appropriate format: patch accepts face vertex indices matrix.
faces = node_map;

% Build patch arguments
pArgs = varargin;

if isNodal
    % Nodal data: use FaceVertexCData with interpolation
    % Create patch with vertices and faces, supply FaceVertexCData
    % Set FaceColor to 'interp' and EdgeColor as provided or 'none' by default
    if ~any(strcmpi('FaceColor',pArgs))
        pArgs = [{'FaceColor','interp'}, pArgs];
    end
    if ~any(strcmpi('EdgeColor',pArgs))
        pArgs = [{'EdgeColor','none'}, pArgs];
    end
    % Create patch
    f = patch('Vertices', verts, 'Faces', faces, 'FaceVertexCData', c, pArgs{:});

else
    % Cell-centered: supply per-face CData and use flat coloring
    if ~any(strcmpi('FaceColor',pArgs))
        pArgs = [{'FaceColor','flat'}, pArgs];
    end
    if ~any(strcmpi('EdgeColor',pArgs))
        pArgs = [{'EdgeColor','none'}, pArgs];
    end
    % patch expects CData as Mx1 or Mx3 color; provide as Mx1
    faceCData = c;
    % Create patch
    f = patch('Vertices', verts, 'Faces', faces, 'CData', faceCData, pArgs{:});

end

% Set colormap and colorbar behavior consistent with contourf
if ~any(strcmpi('EdgeColor',pArgs))
    set(f,'EdgeColor','none');
end
axis equal
if dim == 3
    view(3)
    camlight headlight
    lighting gouraud
end
colorbar
end