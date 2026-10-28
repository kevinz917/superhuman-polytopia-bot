// Independent Microsoft System.Random vectors; no game engine dependency.
using System.Text.Json;
var seeds=new int[]{0,1,7,42,-1,161803399,2147483647,-2147483648};
var cases=seeds.Select(seed=>{
    var raw=new Random(seed); var range=new Random(seed); var floats=new Random(seed);
    return new {seed,samples=Enumerable.Range(0,128).Select(_=>raw.Next()).ToArray(),
        bounded=Enumerable.Range(0,128).Select(_=>range.Next(0,196)).ToArray(),
        unit_float=Enumerable.Range(0,128).Select(_=>(float)floats.NextDouble()).ToArray()};
});
Console.WriteLine(JsonSerializer.Serialize(new {source="Microsoft .NET 8 System.Random(int), Next and NextDouble",cases}));
