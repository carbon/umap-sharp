using System.Linq;
using System.Runtime.CompilerServices;

namespace Carbon.AI.Umap;

internal sealed class SparseMatrix
{
    private readonly Dictionary<RowCol, float> _entries;

    public SparseMatrix(int[] rows, int[] cols, float[] values, (int rows, int cols) dims) : this(Combine(rows, cols, values), dims) { }
    
    private SparseMatrix(IEnumerable<(int row, int col, float value)> entries, (int rows, int cols) dims)
    {
        Dims = dims;
        _entries = [];
        foreach (var (entry, index) in entries.Select((entry, index) => (entry, index)))
        {
            CheckDims(entry.row, entry.col);
            _entries[new RowCol(entry.row, entry.col)] = entry.value;
        }
    }

    private SparseMatrix(Dictionary<RowCol, float> entries, (int, int) dims)
    {
        Dims = dims;
        _entries = entries;
    }

    private static IEnumerable<(int row, int col, float value)> Combine(int[] rows, int[] cols, float[] values)
    {
        var rowsArray = rows;
        var colsArray = cols;
        var valuesArray = values;
        if ((rowsArray.Length != valuesArray.Length) || (colsArray.Length != valuesArray.Length))
        {
            throw new ArgumentException($"The input lists {nameof(rows)}, {nameof(cols)} and {nameof(values)} must all have the same number of elements");
        }

        for (var i = 0; i < valuesArray.Length; i++)
        {
            yield return (rowsArray[i], colsArray[i], valuesArray[i]);
        }
    }

    public (int rows, int cols) Dims { get; }

    public void Set(int row, int col, float value)
    {
        CheckDims(row, col);
        _entries[new RowCol(row, col)] = value;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float Get(int row, int col, float defaultValue = 0)
    {
        CheckDims(row, col);
        return _entries.TryGetValue(new RowCol(row, col), out var v) ? v : defaultValue;
    }

    public IEnumerable<(int row, int col, float value)> GetAll() => _entries.Select(kv => (kv.Key.Row, kv.Key.Col, kv.Value));

    public IEnumerable<int> GetRows() => _entries.Keys.Select(k => k.Row);
    public IEnumerable<int> GetCols() => _entries.Keys.Select(k => k.Col);
    public IEnumerable<float> GetValues() => _entries.Values;
    
    public void ForEach(Action<float, int, int> fn)
    {
        foreach (var kv in _entries)
        {
            fn(kv.Value, kv.Key.Row, kv.Key.Col);
        }
    }

    public SparseMatrix Map(Func<float, float> fn) => Map((value, row, col) => fn(value));

    public SparseMatrix Map(Func<float, int, int, float> fn)
    {
        var newEntries = _entries.ToDictionary(kv => kv.Key, kv => fn(kv.Value, kv.Key.Row, kv.Key.Col));
        return new SparseMatrix(newEntries, Dims);
    }

    public float[][] ToArray()
    {
        var output = Enumerable.Range(0, Dims.rows).Select(_ => new float[Dims.cols]).ToArray();
        foreach (var kv in _entries)
        {
            output[kv.Key.Row][kv.Key.Col] = kv.Value;
        }

        return output;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void CheckDims(int row, int col)
    {
#if DEBUG
        if ((row >= Dims.rows) || (col >= Dims.cols))
        {
            throw new Exception("array index out of bounds");
        }
#endif
    }

    public SparseMatrix Transpose()
    {
        var dims = (Dims.cols, Dims.rows);
        var entries = new Dictionary<RowCol, float>(_entries.Count);
        foreach (var entry in _entries)
        {
            entries[new RowCol(entry.Key.Col, entry.Key.Row)] = entry.Value;
        }

        return new SparseMatrix(entries, dims);
    }

    /// <summary>
    /// Element-wise multiplication of two matrices
    /// </summary>
    public SparseMatrix PairwiseMultiply(SparseMatrix other)
    {
        var newEntries = new Dictionary<RowCol, float>(_entries.Count);
        foreach (var kv in _entries)
        {
            if (other._entries.TryGetValue(kv.Key, out var v))
            {
                newEntries[kv.Key] = kv.Value * v;
            }
        }
        return new SparseMatrix(newEntries, Dims);
    }

    /// <summary>
    /// Element-wise addition of two matrices
    /// </summary>
    public SparseMatrix Add(SparseMatrix other) => ElementWiseWith(other, (x, y) => x + y);

    /// <summary>
    /// Element-wise subtraction of two matrices
    /// </summary>
    public SparseMatrix Subtract(SparseMatrix other) => ElementWiseWith(other, (x, y) => x - y);

    /// <summary>
    /// Scalar multiplication of a matrix
    /// </summary>
    public SparseMatrix MultiplyScalar(float scalar) => Map((value, row, cols) => value * scalar);

    /// <summary>
    /// Helper function for element-wise operations
    /// </summary>
    private SparseMatrix ElementWiseWith(SparseMatrix other, Func<float, float, float> op)
    {
        var newEntries = new Dictionary<RowCol, float>(_entries.Count);
        foreach (var k in _entries.Keys.Union(other._entries.Keys))
        {
            newEntries[k] = op(
                _entries.TryGetValue(k, out var x) ? x : 0f,
                other._entries.TryGetValue(k, out var y) ? y : 0f
            );
        }
        return new SparseMatrix(newEntries, Dims);
    }

    /// <summary>
    /// Helper function for getting data, indices, and indptr arrays from a sparse matrix to follow csr matrix conventions. Super inefficient (and kind of defeats the purpose of this convention)
    /// but a lot of the ported python tree search logic depends on this data format.
    /// </summary>
    public (int[] indices, float[] values, int[] indptr) GetCSR()
    {
        var entries = new List<(float value, int row, int col)>();
        ForEach((value, row, col) => entries.Add((value, row, col)));
        entries.Sort((a, b) =>
        {
            if (a.row == b.row)
            {
                return a.col - b.col;
            }

            return a.row - b.row;
        });

        var indices = new List<int>();
        var values = new List<float>();
        var indptr = new List<int>();
        var currentRow = -1;
        for (var i = 0; i < entries.Count; i++)
        {
            var (value, row, col) = entries[i];
            if (row != currentRow)
            {
                currentRow = row;
                indptr.Add(i);
            }
            indices.Add(col);
            values.Add(value);
        }
        return (indices.ToArray(), values.ToArray(), indptr.ToArray());
    }

    private readonly struct RowCol(int row, int col) : IEquatable<RowCol>
    {
        public int Row { get; } = row;

        public int Col { get; } = col;

        public bool Equals(RowCol other) => (other.Row == Row) && (other.Col == Col);
        
        public override bool Equals(object obj) => obj is RowCol other && Equals(other);

        public override int GetHashCode() => HashCode.Combine(Row, Col);
    }
}