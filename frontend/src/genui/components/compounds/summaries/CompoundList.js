import React from 'react';
import { Col, Row } from 'reactstrap';
import { ActivitySetList, MoleculeActivityDetail, MoleculeData, MoleculeDetail } from '../../..';

export function CompoundListItem(props) {
  const mol = props.mol;
  return (
    <Row>
      <Col md={3} sm={3}>
        <MoleculeDetail mol={mol}/>
      </Col>
      <Col md={3} sm={3}>
        <MoleculeData {...props} mol={mol}/>
      </Col>
      <Col md={6} sm={6}>
        <MoleculeActivityDetail
          {...props}
          mol={mol}
          component={ActivitySetList}
        />
      </Col>
    </Row>
  )
}

export default function CompoundList(props) {
  const mols = props.mols;

  return (
    <React.Fragment>
      {
        mols.map(mol => (
          <CompoundListItem {...props} key={mol.id} mol={mol}/>
        ))
      }
    </React.Fragment>
  )

}