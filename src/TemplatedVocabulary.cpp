#include "DBoW2/TemplatedVocabulary.h"

#include "DBoW2/FBrief.h"
#include "DBoW2/FSurf64.h"
#include "brief_node.pb.h"
#include "brief_vocabulary.pb.h"

namespace DBoW2 {

template<>
void TemplatedVocabulary<DBoW2::FBrief::TDescriptor, DBoW2::FBrief>::loadProto(
    const std::string& file_name)
{
  std::ifstream file(file_name.c_str());
  CHECK(file.is_open()) << "Couldn't open " << file_name;

  google::protobuf::io::IstreamInputStream raw_in(&file);
  google::protobuf::io::CodedInputStream coded_in(&raw_in);
  coded_in.SetTotalBytesLimit(std::numeric_limits<int>::max(), -1);

  proto::BriefVocabulary vocabulary_proto;
  CHECK(vocabulary_proto.ParseFromCodedStream(&coded_in));

  m_k = vocabulary_proto.k();
  m_L = vocabulary_proto.l();
  m_scoring = static_cast<ScoringType>(vocabulary_proto.scoring_type());
  m_weighting = static_cast<WeightingType>(vocabulary_proto.weighting_type());

  createScoringObject();

  // nodes
  m_nodes.resize(vocabulary_proto.nodes_size() + 1); // +1 to include root
  m_nodes[0].id = 0;

  for(const proto::BriefNode& node : vocabulary_proto.nodes())
  {
    const NodeId node_id = node.node_id();
    const NodeId parent_id = node.parent_id();

    m_nodes[node_id].id = node.node_id();
    m_nodes[node_id].parent = node.parent_id();
    m_nodes[node_id].weight = node.weight();

    m_nodes[parent_id].children.push_back(node_id);

    // For now only works with BRIEF.
    std::vector<unsigned long> descriptor_blocks;
    descriptor_blocks.reserve(node.node_descriptor_size());
    for (const uint64_t block : node.node_descriptor())
    {
      descriptor_blocks.emplace_back(block);
    }

    m_nodes[node_id].descriptor.append(
        descriptor_blocks.begin(), descriptor_blocks.end());
  }

  m_words.resize(vocabulary_proto.words_size());

  for(const proto::BriefWord& word : vocabulary_proto.words())
  {
    const WordId word_id = word.word_id();
    const NodeId node_id = word.node_id();

    m_nodes[node_id].word_id = word_id;
    m_words[word_id] = &m_nodes[node_id];
  }
}

template<>
void TemplatedVocabulary<
DBoW2::FSurf64::TDescriptor, DBoW2::FSurf64>::loadProto(
    const std::string& file_name)
{
  LOG(FATAL) << "Not implemented yet!";
}

template<>
void TemplatedVocabulary<DBoW2::FBrief::TDescriptor, DBoW2::FBrief>::saveProto(
    const std::string& file_name) const
{
  std::ofstream file(file_name);
  CHECK(file.is_open()) << "Couldn't open " << file_name;
  proto::BriefVocabulary vocabulary_proto;

  vocabulary_proto.set_k(m_k);
  vocabulary_proto.set_l(m_L);
  vocabulary_proto.set_scoring_type(m_scoring);
  vocabulary_proto.set_weighting_type(m_weighting);

  vector<NodeId> parents, children;
  vector<NodeId>::const_iterator pit;

  parents.push_back(0); // root

  while(!parents.empty())
  {
    NodeId pid = parents.back();
    parents.pop_back();

    const Node& parent = m_nodes[pid];
    children = parent.children;

    for(pit = children.begin(); pit != children.end(); pit++)
    {
      const Node& child = m_nodes[*pit];

      proto::BriefNode* node_proto = vocabulary_proto.add_nodes();
      CHECK_NOTNULL(node_proto);
      node_proto->set_node_id(child.id);
      node_proto->set_parent_id(pid);
      node_proto->set_weight(child.weight);

      // For now only works with BRIEF.
      std::vector<unsigned long> descriptor_blocks(
          child.descriptor.num_blocks());
      boost::to_block_range(child.descriptor, descriptor_blocks.begin());

      for (const unsigned long block : descriptor_blocks)
      {
        node_proto->add_node_descriptor(block);
      }

      // add to parent list
      if(!child.isLeaf())
      {
        parents.push_back(*pit);
      }
    }
  }

  typename vector<Node*>::const_iterator wit;
  for(wit = m_words.begin(); wit != m_words.end(); wit++)
  {
    WordId id = wit - m_words.begin();

    proto::BriefWord* word_proto = vocabulary_proto.add_words();
    CHECK_NOTNULL(word_proto);
    word_proto->set_word_id(id);
    word_proto->set_node_id((*wit)->id);
  }

  CHECK(vocabulary_proto.SerializeToOstream(&file));
}

}  // namespace DBoW2
