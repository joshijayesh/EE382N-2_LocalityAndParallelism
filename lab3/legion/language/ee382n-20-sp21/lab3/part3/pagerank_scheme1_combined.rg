import "regent"

-- Helper module to handle command line arguments
local PageRankConfig = require("pagerank_config")

local c = regentlib.c

fspace Page {
  rank         : double,
  delta        : double,
  outgoing_links: int64,
  old_rank : double,
  residue :  double
}

--
-- TODO: Define fieldspace 'Link' which has two pointer fields,
--       one that points to the source and another to the destination.
--
fspace Link(r: region(Page))
{
  source_page : ptr(Page, r),
  dest_page : ptr(Page, r)

 }

fspace norm
{
  value: double
}
terra skip_header(f : &c.FILE)
  var x : uint64, y : uint64
  c.fscanf(f, "%llu\n%llu\n", &x, &y)
end

terra read_ids(f : &c.FILE, page_ids : &uint32)
  return c.fscanf(f, "%d %d\n", &page_ids[0], &page_ids[1]) == 2
end

task initialize_graph(r_pages   : region(Page),
                      --
                      -- TODO: Give the right region type here.
                      --
                      r_links   : region(Link(r_pages)),
                      damp      : double,
                      num_pages : uint64,
                      filename  : int8[512])
where
  reads writes(r_pages, r_links)
do
  var ts_start = c.legion_get_current_time_in_micros()
  for page in r_pages do
    page.rank = 1.0 / num_pages
    page.delta = 0.0
    page.residue = 0.0
    -- TODO: Initialize your fields if you need
  end

  var f = c.fopen(filename, "rb")
  skip_header(f)
  var page_ids : uint32[2]
  for link in r_links do
    regentlib.assert(read_ids(f, page_ids), "Less data that it should be")
    var src_page = dynamic_cast(ptr(Page, r_pages), page_ids[0])
    var dst_page = dynamic_cast(ptr(Page, r_pages), page_ids[1])
    --
    -- TODO: Initialize the link with 'src_page' and 'dst_page'
    --
    link.source_page = src_page
    link.dest_page = dst_page
    src_page.outgoing_links += 1
  end

  for page in r_pages do
    page.delta = page.rank/page.outgoing_links
  end
  c.fclose(f)
  var ts_stop = c.legion_get_current_time_in_micros()
  c.printf("Graph initialization took %.4f sec\n", (ts_stop - ts_start) * 1e-6)
end

task Delta_update(r_pages   : region(Page),
                   damp : double,
                   num_pages : uint64)
where
  reads(r_pages.rank, r_pages.delta, r_pages.outgoing_links),
  writes(r_pages.delta, r_pages.old_rank, r_pages.rank)
do
for page in r_pages do
    page.delta = page.rank/page.outgoing_links
    page.old_rank = page.rank
    page.rank = (1.0-damp)/num_pages
end
end

task PageRank_update(r_pages   : region(Page),
                      r_links   : region(Link(r_pages)),
                      damp      : double,
                      num_pages : uint64)
where
  reads(r_links, r_pages.rank, r_pages.delta, r_pages.outgoing_links),
  writes(r_pages.rank, r_pages.delta, r_pages.old_rank)
do
  for page in r_pages do
      -- page.delta = page.rank/page.outgoing_links
      page.old_rank = page.rank
      page.rank = (1.0-damp)/num_pages
  end

  for link in r_links do
    link.dest_page.rank += damp*link.source_page.delta
  end
end

task dump_ranks(r_pages  : region(Page),

                filename : int8[512])
where
  reads(r_pages.rank)
do
  var f = c.fopen(filename, "w")
  for page in r_pages do c.fprintf(f, "%g\n", page.rank) end
  c.fclose(f)
end

task norm_partition(r_pages : region(Page))
where
  reads(r_pages.rank, r_pages.old_rank, r_pages.outgoing_links),
  writes(r_pages.delta)
do
  var l2_norm = 0.0
  for page in r_pages do
    var change = (page.rank-page.old_rank)
    l2_norm += change*change
    page.delta = page.rank/page.outgoing_links
  end
  return l2_norm
end

task toplevel()
  var config : PageRankConfig
  config:initialize_from_command()
  c.printf("**********************************\n")
  c.printf("* PageRank                       *\n")
  c.printf("*                                *\n")
  c.printf("* Number of Pages  : %11lu *\n",  config.num_pages)
  c.printf("* Number of Links  : %11lu *\n",  config.num_links)
  c.printf("* Damping Factor   : %11.4f *\n", config.damp)
  c.printf("* Error Bound      : %11g *\n",   config.error_bound)
  c.printf("* Max # Iterations : %11u *\n",   config.max_iterations)
  c.printf("* # Parallel Tasks : %11u *\n",   config.parallelism)
  c.printf("**********************************\n")

  -- Create a region of pages
  var r_pages = region(ispace(ptr, config.num_pages), Page)
  var r_links = region(ispace(ptr, config.num_links), Link(wild))

  -- Initialize the page graph from a file
  initialize_graph(r_pages,r_links, config.damp, config.num_pages, config.input)
  
  var page_partition = partition(equal,r_pages,ispace(int1d, config.parallelism))
  var link_partition = preimage(disjoint,complete,r_links,page_partition,r_links.dest_page)
  var r_norm = region(ispace(int1d, config.parallelism),norm)
  
  var num_iterations = 0
  var converged = false
  __fence(__execution, __block) -- This blocks to make sure we only time the pagerank computation
  var ts_start = c.legion_get_current_time_in_micros()
  while not converged do
    num_iterations += 1
    
    --for c in page_partition.colors
    --[[
    for c in page_partition.colors do
      Delta_update(page_partition[c], config.damp, config.num_pages)
    end
    --]]
  
    for c in link_partition.colors do
      PageRank_update(page_partition[c],link_partition[c],config.damp, config.num_pages)
    end

    for c in page_partition.colors do
      r_norm[c].value = norm_partition(page_partition[c])--,config.error_bound)
    end
     
    var l2_norm = 0.0
    for x in r_norm do
      l2_norm += x.value
    end    
     
    converged = l2_norm <= config.error_bound*config.error_bound

    if num_iterations > config.max_iterations then
      break
    end
  end
  __fence(__execution, __block) -- This blocks to make sure we only time the pagerank computation
  var ts_stop = c.legion_get_current_time_in_micros()
  c.printf("PageRank converged after %d iterations in %.4f sec\n",
    num_iterations, (ts_stop - ts_start) * 1e-6)

  if config.dump_output then dump_ranks(r_pages, config.output) end
end

regentlib.start(toplevel)

