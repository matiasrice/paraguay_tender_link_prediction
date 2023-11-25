# paraguay_tender_link_prediction

In many countries, the government stands as one of the largest entities engaging
in the procurement of services and the purchase of products. The initiation of this
procurement phase typically involves a tendering process. Unfortunately, there
are instances where the number of companies participating in these processes is
limited. This challenge could be addressed by implementing a strategy wherein
companies already providing services to other government institutions are made
aware of additional tender opportunities. This approach can be conceptualized
as a link prediction problem, where the goal is to forecast future or plausible
links between a supplier and an institution. To operationalize this concept, we
extract features from the public tender information in Paraguay and represent the
supplier-institution relationship using a bipartite graph structure. We harness the
capabilities of two contemporary models, GraphSAGE and GAT, to enhance the
classification task. The experimental results demonstrate a high level of precision
following the application of these methods.
