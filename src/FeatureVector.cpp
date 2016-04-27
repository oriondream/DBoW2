/**
 * File: FeatureVector.cpp
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: feature vector
 * License: see the LICENSE.txt file
 *
 */

#include <map>
#include <vector>
#include <iostream>

#include <glog/logging.h>

#include "DBoW2/FeatureVector.h"

namespace DBoW2 {

// ---------------------------------------------------------------------------

FeatureVector::FeatureVector(void)
{
}

// ---------------------------------------------------------------------------

FeatureVector::~FeatureVector(void)
{
}

// ---------------------------------------------------------------------------

void FeatureVector::addFeature(NodeId id, unsigned int i_feature)
{
  FeatureVector::iterator vit = this->lower_bound(id);
  
  if(vit != this->end() && vit->first == id)
  {
    vit->second.push_back(i_feature);
  }
  else
  {
    vit = this->insert(vit, FeatureVector::value_type(id, 
      std::vector<unsigned int>() ));
    vit->second.push_back(i_feature);
  }
}

// ---------------------------------------------------------------------------

std::ostream& operator<<(std::ostream &out, 
  const FeatureVector &v)
{
  if(!v.empty())
  {
    FeatureVector::const_iterator vit = v.begin();
    
    const std::vector<unsigned int>* f = &vit->second;

    out << "<" << vit->first << ": [";
    if(!f->empty()) out << (*f)[0];
    for(unsigned int i = 1; i < f->size(); ++i)
    {
      out << ", " << (*f)[i];
    }
    out << "]>";
    
    for(++vit; vit != v.end(); ++vit)
    {
      f = &vit->second;
      
      out << ", <" << vit->first << ": [";
      if(!f->empty()) out << (*f)[0];
      for(unsigned int i = 1; i < f->size(); ++i)
      {
        out << ", " << (*f)[i];
      }
      out << "]>";
    }
  }
  
  return out;  
}

void FeatureVector::filter(const std::vector<unsigned int>& remaining_indices)
{
  typedef std::map<unsigned long, NodeId> InverseMap;
  InverseMap inverse_map;
  for (const value_type& node_indices : *this)
  {
    for (const unsigned int index : node_indices.second)
    {
      CHECK(inverse_map.emplace(index, node_indices.first).second);
    }
  }

  FeatureVector filtered;
  for (const unsigned int index : remaining_indices)
  {
    InverseMap::iterator found = inverse_map.find(index);
    CHECK(found != inverse_map.end());
    filtered.addFeature(found->second, found->first);
  }

  swap(filtered);
}

void FeatureVector::forEachCommonNode(
      const FeatureVector& other, const std::function<void(
          const NodeId& node_id, const std::vector<unsigned int>& own_indices,
          const std::vector<unsigned int>& other_indices)>& action) const
{
  CHECK(action);
  const_iterator own_it = begin();
  const_iterator other_it = other.begin();

  while (own_it != end() && other_it != other.end())
  {
    if (own_it->first == other_it->first)
    {
      action(own_it->first, own_it->second, other_it->second);
      ++own_it;
      ++other_it;
    }
    else if (own_it->first < other_it->first)
    {
      own_it = lower_bound(other_it->first);
    }
    else
    {
      other_it = other.lower_bound(own_it->first);
    }
  }
}

// ---------------------------------------------------------------------------

} // namespace DBoW2
